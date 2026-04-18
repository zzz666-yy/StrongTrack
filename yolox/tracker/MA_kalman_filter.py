# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class MAkalmanFilter(object):
    """

    This filter dynamically adjusts process and observation covariance matrices based on:
    1. Motion Matching Cost: Adapts process noise when motion predictions are inaccurate
    2. Detection Confidence: Adapts observation noise based on detection quality

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.alpha = 25.0
        self.beta = 0.5
        self.gamma = 2.0

        # Adaptive thresholds
        self.high_confidence_thresh = 0.7
        self.low_confidence_thresh = 0.3
        self.high_motion_cost_thresh = 0.5

        # Base noise levels
        self.base_process_noise = 1.0
        self.base_observation_noise = 1.0

        # Lambda_k parameters (appearance-aware observation adaptation)
        self.lambda_alpha = 0.4
        self.lambda_beta = 0.8
        self.lambda_gamma = 0.8
        self.lambda_min = 0.4
        self.lambda_max = 1.0
        self.lambda_eps = 1e-3
        self.lambda_rho = 0.8  # Exponential smoothing factor
        self.min_observation_noise = 0.2
        self.max_observation_noise = 10.0

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),测量是一个4维数组
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        k = np.asarray([1])
        return mean, covariance, k

    def predict(self, mean, covariance, k):
        """Run Kalman filter prediction step with adaptive process noise.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        k : float or ndarray
            Motion adaptation factor based on previous IoU cost

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state with adaptive process noise.

        """
        # Ensure k is a scalar
        if isinstance(k, np.ndarray):
            k = k.item() if k.size == 1 else k[0]

        # Adaptive process noise based on motion matching cost
        # Higher k means higher motion cost, so increase process noise
        motion_adaptation = self.base_process_noise * (1.0 + self.gamma * k)

        std_pos = [
            self._std_weight_position * mean[3] * motion_adaptation,
            self._std_weight_position * mean[3] * motion_adaptation,
            1e-2 * motion_adaptation,
            self._std_weight_position * mean[3] * motion_adaptation]
        std_vel = [
            self._std_weight_velocity * mean[3] * motion_adaptation,
            self._std_weight_velocity * mean[3] * motion_adaptation,
            1e-5 * motion_adaptation,
            self._std_weight_velocity * mean[3] * motion_adaptation]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, scores, lambda_factor=1.0, confidence=0.0):
        """Project state distribution to measurement space with adaptive observation noise.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        scores : float
            Detection confidence score (0-1)
        confidence : float
            Additional confidence factor (legacy parameter)

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix with adaptive
            observation noise based on detection confidence.

        """
        # Adaptive observation noise based on detection confidence
        # Lower confidence -> higher observation noise (less trust in measurement)
        confidence_adaptation = self._compute_confidence_adaptation(scores, lambda_factor)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # Apply confidence-based adaptation
        std = [confidence_adaptation * x for x in std]

        # Additional legacy confidence factor
        if confidence > 0:
            std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def _compute_confidence_adaptation(self, scores, lambda_factor):
        """Compute observation noise adaptation factor based on detection confidence.

        Parameters
        ----------
        scores : float
            Detection confidence score (0-1)

        Returns
        -------
        float
            Adaptation factor for observation noise
        """
        # Ensure scores is a scalar
        if isinstance(scores, np.ndarray):
            scores = scores.item() if scores.size == 1 else scores[0]

        # Clamp scores to valid range
        scores = np.clip(scores, 0.0, 1.0)

        # Adaptive factor logic
        if scores >= self.high_confidence_thresh:
            # High confidence: reduce noise
            adaptation_factor = self.base_observation_noise * (
                    1.0 - self.beta * (scores - self.high_confidence_thresh)
            )
        elif scores > self.low_confidence_thresh:  # 注意这里用 >
            # Medium confidence: moderate increase
            adaptation_factor = self.base_observation_noise * (
                    1.0 + self.beta * (self.high_confidence_thresh - scores)
            )
        else:
            # Low confidence: Significant increase (Cumulative)
            # 1. Calculate the max penalty from the medium phase
            medium_penalty_max = self.beta * (self.high_confidence_thresh - self.low_confidence_thresh)

            # 2. Calculate extra penalty for low phase
            low_penalty_extra = 2.0 * self.beta * (self.low_confidence_thresh - scores)

            # 3. Sum them up
            adaptation_factor = self.base_observation_noise * (
                    1.0 + medium_penalty_max + low_penalty_extra
            )

        adaptation_factor *= lambda_factor
        return float(np.clip(adaptation_factor, self.min_observation_noise, self.max_observation_noise))

    def multi_predict(self, mean, covariance, k):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        k : ndarray
            the N*1 of previous step
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """

        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        a = np.r_[std_pos, std_vel]
        sqr = np.square(a).T

        motion_cov = []
        for i in range(len(mean)):
            b = np.diag(sqr[i])
            ik = k[i]
            motion_cov.append(ik * b)
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, scores, cost_IOU_measurement,
               reid_similarity=None, prev_lambda=1.0):
        """Run Kalman filter correction step with adaptive parameters.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        scores : float
            Detection confidence score (0-1)
        cost_IOU_measurement : float
            IoU-based matching cost (0-1, where 0 is perfect match)
        reid_similarity : float, optional
            Appearance similarity score (0-1)
        prev_lambda : float, optional
            Smoothed lambda from previous frame for temporal smoothing.

        Returns
        -------
        (ndarray, ndarray, ndarray, float)
            Returns the measurement-corrected state distribution, updated
            motion adaptation factor, and smoothed lambda value.

        """
        lambda_k = self._compute_lambda(cost_IOU_measurement, scores, reid_similarity)
        lambda_bar = self._smooth_lambda(lambda_k, prev_lambda)

        # Project state to measurement space with confidence-adaptive noise
        projected_mean, projected_cov = self.project(mean, covariance, scores, lambda_bar)

        # Compute Kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        # Update state estimate
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        # Compute motion adaptation factor for next prediction
        new_k = self._compute_motion_adaptation(cost_IOU_measurement, scores)

        return new_mean, new_covariance, new_k, lambda_bar

    def _compute_motion_adaptation(self, cost_IOU_measurement, scores):
        """Compute motion adaptation factor based on IoU cost and detection confidence.

        Parameters
        ----------
        cost_IOU_measurement : float
            IoU-based matching cost (0-1, where 0 is perfect match)
        scores : float
            Detection confidence score (0-1)

        Returns
        -------
        ndarray
            Motion adaptation factor for next prediction step
        """
        # Ensure inputs are scalars
        if isinstance(cost_IOU_measurement, np.ndarray):
            cost_IOU_measurement = cost_IOU_measurement.item() if cost_IOU_measurement.size == 1 else \
            cost_IOU_measurement[0]
        if isinstance(scores, np.ndarray):
            scores = scores.item() if scores.size == 1 else scores[0]

        # Clamp inputs to valid ranges
        cost_IOU_measurement = np.clip(cost_IOU_measurement, 0.0, 1.0)
        scores = np.clip(scores, 0.0, 1.0)

        # Convert IoU cost to IoU (higher IoU = better match)
        iou = 1.0 - cost_IOU_measurement

        # Motion adaptation based on IoU quality and detection confidence
        # Poor IoU or low confidence -> higher adaptation factor -> more process noise
        iou_factor = 1.0 - iou ** self.alpha
        confidence_factor = 1.0 - scores ** (self.beta * 10)  # Scale beta for motion adaptation

        # Combine factors: both poor IoU and low confidence increase adaptation
        motion_adaptation = max(iou_factor, confidence_factor)

        # Additional penalty for very poor matches
        if cost_IOU_measurement > self.high_motion_cost_thresh:
            motion_adaptation *= 1.5

        return np.asarray([motion_adaptation])

    def _compute_lambda(self, cost_IOU_measurement, scores, reid_similarity):
        """Compute instantaneous λ_k combining geometry, confidence and appearance."""
        if isinstance(cost_IOU_measurement, np.ndarray):
            cost_IOU_measurement = cost_IOU_measurement.item() if cost_IOU_measurement.size == 1 else cost_IOU_measurement[0]
        if isinstance(scores, np.ndarray):
            scores = scores.item() if scores.size == 1 else scores[0]
        if isinstance(reid_similarity, np.ndarray):
            reid_similarity = reid_similarity.item() if reid_similarity.size == 1 else reid_similarity[0]

        d = float(np.clip(cost_IOU_measurement if cost_IOU_measurement is not None else 0.5,
                          self.lambda_eps, 1.0))
        c = float(np.clip(scores if scores is not None else 0.5, self.lambda_eps, 1.0))
        if reid_similarity is None or reid_similarity < 0.5:
            s = 1.0
        else:
            s = float(np.clip(reid_similarity, self.lambda_eps, 1.0))

        lambda_raw = ((d ** self.lambda_alpha) / ((c * s) ** self.lambda_beta)) ** self.lambda_gamma
        return float(np.clip(lambda_raw, self.lambda_min, self.lambda_max))

    def _smooth_lambda(self, lambda_k, prev_lambda):
        """Exponentially smooth λ_k to stabilize observation covariance."""
        if isinstance(prev_lambda, np.ndarray):
            prev_lambda = prev_lambda.item() if prev_lambda.size == 1 else prev_lambda[0]

        if prev_lambda is None or not np.isfinite(prev_lambda):
            smoothed = lambda_k
        else:
            smoothed = self.lambda_rho * prev_lambda + (1.0 - self.lambda_rho) * lambda_k

        return float(np.clip(smoothed, self.lambda_min, self.lambda_max))

