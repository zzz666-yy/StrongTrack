import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import cv2
from .gmc import GMC

from .MA_kalman_filter import MAkalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from yolox.data.dataloading import get_yolox_datadir
try:
    from fast_reid.fast_reid_interfece import FastReIDInterface
except Exception:
    FastReIDInterface = None

class STrack(BaseTrack):
    shared_kalman = MAkalmanFilter()
    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.occluded = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat, score=-1):
        """
        Improved: Confidence-Weighted Feature Update
        References: BoT-SORT (arXiv:2206.14651) logic for feature updating.
        """
        if feat is None:
            return
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        self.curr_feat = feat

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:

            if score > 0:

                update_rate = (1.0 - self.alpha) * score
                current_alpha = 1.0 - update_rate
            else:
                current_alpha = self.alpha

            self.smooth_feat = current_alpha * self.smooth_feat + (1.0 - current_alpha) * feat

        self.features.append(feat)
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-12)

        # PA Kalman Filter specific (keep original)
        self.k = None
        self.cost_IOU = 0.0
        self.lambda_bar = 1.0

    def _compute_reid_similarity(self, detection):
        """
        Improved: Gallery-based Matching (Max-Cosine Similarity)
        Compute the maximum similarity between the detection and the track's feature history.
        """
        if detection is None:
            return None


        det_feat = getattr(detection, 'curr_feat', None)
        if det_feat is None:
            det_feat = getattr(detection, 'smooth_feat', None)

        if det_feat is None:
            return None


        base_feat = getattr(self, 'smooth_feat', None)
        sim_smooth = -1.0
        if base_feat is not None:
            sim_smooth = np.dot(base_feat, det_feat)
        sim_history = -1.0
        if len(self.features) > 0:
            history_feats = np.array(self.features)  # Shape: (N, Feature_Dim)
            sims = np.dot(history_feats, det_feat)
            sim_history = np.max(sims)
        cosine_sim = max(sim_smooth, sim_history)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        similarity = 0.5 * (cosine_sim + 1.0)

        # Treat very low similarity as unreliable
        if similarity < 0.2:
            return None

        return similarity

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        if isinstance(self.kalman_filter, MAkalmanFilter) and self.k is not None:
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, self.k.copy())
        else:
            self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            if isinstance(STrack.shared_kalman, MAkalmanFilter):
                multi_k = np.asarray([st.k if st.k is not None else np.asarray([1]) for st in stracks], dtype=object)
                for i, st in enumerate(stracks):
                    if st.state != TrackState.Tracked:
                        multi_mean[i][7] = 0
                multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance, multi_k)
            else:
                for i, st in enumerate(stracks):
                    if st.state != TrackState.Tracked:
                        multi_mean[i][7] = 0
                multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """
        Apply camera motion compensation (2x3 affine) to a list of tracks.
        This implementation is adapted for ByteTrack's 8D state: [x, y, a, h, vx, vy, va, vh]
        - Apply R,t to position (x,y) and R to velocity (vx,vy)
        - Keep (a, h, va, vh) unchanged
        - Propagate covariance with corresponding linear transform T (8x8)
        """
        if len(stracks) == 0:
            return
        R = H[:2, :2]
        t = H[:2, 2]
        # Build 8x8 transform matrix for mean/covariance
        T = np.eye(8, dtype=float)
        T[0:2, 0:2] = R  # position
        T[4:6, 4:6] = R  # velocity
        for st in stracks:
            if st.mean is None or st.covariance is None:
                continue
            # Update mean: position and velocity
            pos = st.mean[0:2]
            vel = st.mean[4:6]
            st.mean[0:2] = R.dot(pos) + t
            st.mean[4:6] = R.dot(vel)
            # Update covariance
            st.covariance = T.dot(st.covariance).dot(T.T)

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        # Handle PA Kalman Filter initialization
        if isinstance(self.kalman_filter, MAkalmanFilter):
            self.mean, self.covariance, self.k = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        else:
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
            self.k = np.asarray([1])
        self.lambda_bar = 1.0

        self.tracklet_len = 0
        self.occluded = False
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, cost_IOU=0.0,new_id=False):
        self.score = new_track.score
        self.cost_IOU = cost_IOU

        reid_similarity = self._compute_reid_similarity(new_track)

        # Handle PA Kalman Filter update
        if isinstance(self.kalman_filter, MAkalmanFilter):
            self.mean, self.covariance, self.k, self.lambda_bar = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh),
                self.score, self.cost_IOU, reid_similarity=reid_similarity, prev_lambda=self.lambda_bar
            )
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
            )

        self.tracklet_len = 0
        self.occluded = False
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if getattr(new_track, 'curr_feat', None) is not None:
            self.update_features(new_track.curr_feat, score=self.score)

    def update(self, new_track, frame_id,cost_IOU=0.0):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.score = new_track.score
        self.cost_IOU = cost_IOU

        reid_similarity = self._compute_reid_similarity(new_track)

        if isinstance(self.kalman_filter, MAkalmanFilter):
            self.mean, self.covariance, self.k, self.lambda_bar = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh),
                self.score, self.cost_IOU, reid_similarity=reid_similarity, prev_lambda=self.lambda_bar)
        else:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True
        self.occluded = False

        self.score = new_track.score
        if getattr(new_track, 'curr_feat', None) is not None:
            self.update_features(new_track.curr_feat, score=self.score)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = MAkalmanFilter()
        self.occlusion_buffer_factor = float(getattr(args, "occlusion_buffer_factor", 1.5))
        self.occlusion_iou_thresh = float(getattr(args, "occlusion_iou_thresh", 0.3))
        self.occlusion_border_ratio = float(getattr(args, "occlusion_border_ratio", 0.1))
        self.with_reid = bool(getattr(args, 'with_reid', False))
        self.proximity_thresh = float(getattr(args, 'proximity_thresh', 0.5))
        self.appearance_thresh = float(getattr(args, 'appearance_thresh', 0.25))
        self.encoder = None
        if self.with_reid and FastReIDInterface is not None:
            cfg_path = getattr(args, 'fast_reid_config', None)
            w_path = getattr(args, 'fast_reid_weights', None)
            if cfg_path and w_path:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                try:
                    self.encoder = FastReIDInterface(cfg_path, w_path, device)
                except Exception:
                    self.encoder = None
            else:
                self.with_reid = False


        # Initialize GMC (camera motion compensation)
        cmc_method = getattr(self.args, "cmc_method", "none")
        seq_name = getattr(self.args, "name", "Seq")
        # Prefer cmc_ablation flag if provided; fallback to generic ablation
        ablation = getattr(self.args, "cmc_ablation", getattr(self.args, "ablation", False))
        try:
            self.gmc = GMC(method=cmc_method, verbose=[seq_name, ablation])
        except Exception:
            # Fallback to no compensation if initialization fails
            self.gmc = GMC(method='none')

    def _is_occluded(self, track, dets_all, img_h, img_w):
        if dets_all is None:
            return False
        if isinstance(dets_all, list):
            if len(dets_all) == 0:
                return False
            dets_all = np.asarray(dets_all, dtype=float)
        if not isinstance(dets_all, np.ndarray):
            return False
        if dets_all.size == 0:
            return False
        tlbr = track.tlbr
        cx = (tlbr[0] + tlbr[2]) / 2.0
        cy = (tlbr[1] + tlbr[3]) / 2.0
        border_x = img_w * self.occlusion_border_ratio
        border_y = img_h * self.occlusion_border_ratio
        if cx < border_x or cx > img_w - border_x or cy < border_y or cy > img_h - border_y:
            return False
        ious = matching.ious(np.asarray([tlbr], dtype=float), dets_all)[0]
        if ious.size == 0:
            return False
        if np.max(ious) >= self.occlusion_iou_thresh:
            return True
        return False

    def update(self, output_results, img_info, img_size, raw_img=None, img_source=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        if len(dets) > 0 and len(dets_second) > 0:
            dets_all = np.concatenate((dets, dets_second), axis=0)
        elif len(dets) > 0:
            dets_all = dets
        elif len(dets_second) > 0:
            dets_all = dets_second
        else:
            dets_all = None

        if len(dets) > 0:
            if self.with_reid and self.encoder is not None and img_source is not None:
                if isinstance(img_source, str):
                    img_path = os.path.join(get_yolox_datadir(), 'mot', 'train', img_source)
                    image = cv2.imread(img_path)
                else:
                    image = img_source
                features_keep = self.encoder.inference(image, dets)
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # Camera Motion Compensation (apply after prediction, before association)
        try:
            warp = np.eye(2, 3, dtype=float)
            if hasattr(self, 'gmc') and self.gmc is not None:
                cmc_method = getattr(self.args, 'cmc_method', 'none')
                if cmc_method == 'file':
                    # Optionally skip first-frame consumption to keep alignment with files that start at f1->f2
                    if getattr(self.args, 'cmc_skip_first', False) and self.frame_id == 1:
                        warp = np.eye(2, 3, dtype=float)
                    else:
                        # Consume one line per frame from GMC file starting at frame 1
                        warp = self.gmc.apply(None, dets)
                else:
                    if raw_img is not None:
                        warp = self.gmc.apply(raw_img, dets)
            # Optional inversion if GMC file/method provides opposite direction
            if getattr(self.args, 'cmc_inverse', False):
                try:
                    warp = cv2.invertAffineTransform(warp.astype(np.float64)).astype(np.float32)
                except Exception:
                    pass
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)
        except Exception:
            # In case GMC fails, continue without compensation
            pass
        dists = matching.iou_distance(strack_pool, detections)
        geom_dists = dists.copy()
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.with_reid and self.encoder is not None and len(strack_pool) > 0 and len(detections) > 0:
            if all(getattr(t, 'smooth_feat', None) is not None for t in strack_pool):
                ious_dists_mask = (dists > self.proximity_thresh)
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(dists, emb_dists)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            cost_iou = geom_dists[itracked, idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id,cost_iou)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, cost_iou,new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        geom_dists_second = dists.copy()
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            cost_iou = geom_dists_second[itracked, idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, cost_iou)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id,cost_iou, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                if self._is_occluded(track, dets_all, img_h, img_w):
                    track.occluded = True
                else:
                    track.occluded = False
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        geom_dists_unconfirmed = dists.copy()
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        if self.with_reid and self.encoder is not None and len(unconfirmed) > 0 and len(detections) > 0:
            if all(getattr(t, 'smooth_feat', None) is not None for t in unconfirmed):
                ious_dists_mask = (dists > self.proximity_thresh)
                emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(dists, emb_dists)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            cost_iou = geom_dists_unconfirmed[itracked, idet]
            unconfirmed[itracked].update(detections[idet], self.frame_id,cost_iou)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            max_time_lost = self.max_time_lost
            if getattr(track, "occluded", False):
                max_time_lost = int(self.max_time_lost * self.occlusion_buffer_factor)
            if self.frame_id - track.end_frame > max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
