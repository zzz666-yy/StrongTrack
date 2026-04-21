import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import os.path as osp
import threading
import time
import traceback
from dataclasses import dataclass

import cv2
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

# =========================
# 固定模型配置（后台静默加载）
# =========================
EXP_FILE = "D:/MOT/ByteTrack-qt/exps/example/mot/yolox_x_ablation.py"
CKPT_FILE = "D:/MOT/ByteTrack-qt/pretrained/bytetrack_x_mot17.pth.tar"
DEVICE = "gpu"
SAVE_RESULT_DEFAULT = False
CAMERA_ID = 0
USERS_FILE = "users.json"
WINDOW_TITLE = "多目标跟踪软件系统"

HELP_TEXT = """多目标跟踪软件使用说明

1. 首次使用请先注册账号，然后登录系统。
2. 点击“调用摄像头”可直接启动本地摄像头进行实时多目标跟踪。
3. 点击“选择跟踪视频”可以导入本地视频文件。
4. 跟踪过程中，界面会实时显示：
   - 当前跟踪画面
   - 实时帧率 FPS
   - 当前画面中的目标数量
   - 当前运行状态
5. 左上角“帮助”按钮可随时查看本说明。
"""

DISPLAY_EVERY_N_FRAMES = 1


@dataclass
class TrackStats:
    fps: float = 0.0
    object_count: int = 0
    frame_id: int = 0


class UserManager:
    def __init__(self, db_path=USERS_FILE):
        self.db_path = db_path
        if not osp.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def _load(self):
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, users):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

    def register(self, username, password):
        username = username.strip()
        if not username or not password:
            return False, "用户名和密码不能为空"
        users = self._load()
        if username in users:
            return False, "用户名已存在"
        users[username] = {
            "password": password,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save(users)
        return True, "注册成功"

    def login(self, username, password):
        username = username.strip()
        users = self._load()
        if username not in users:
            return False, "用户不存在"
        if users[username]["password"] != password:
            return False, "密码错误"
        return True, "登录成功"


class Predictor:
    def __init__(self, model, exp, trt_file=None, decoder=None, device=torch.device("cpu"), fp16=False):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["file_name"] = None
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, img_info


class TrackingEngine:
    def __init__(self, args):
        self.args = args
        self.exp = get_exp(self.args.exp_file, None)
        self.output_dir, self.vis_folder = self._prepare_output_dirs()
        self.predictor = self._build_predictor()
        self.stop_event = threading.Event()

    def _prepare_output_dirs(self):
        experiment_name = getattr(self.exp, "exp_name", "default_exp")
        output_dir = osp.join(self.exp.output_dir, experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
        return output_dir, vis_folder

    def _build_predictor(self):
        device_name = self.args.device
        if device_name == "gpu" and not torch.cuda.is_available():
            logger.warning("未检测到 CUDA，自动切换为 CPU")
            device_name = "cpu"

        device = torch.device("cuda" if device_name == "gpu" else "cpu")
        model = self.exp.get_model().to(device)
        model.eval()

        ckpt_file = self.args.ckpt
        if not osp.exists(ckpt_file):
            raise FileNotFoundError(f"未找到权重文件：{ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        if getattr(self.args, "fuse", False):
            model = fuse_model(model)

        if getattr(self.args, "fp16", False) and device.type == "cuda":
            model = model.half()

        return Predictor(model, self.exp, None, None, device, getattr(self.args, "fp16", False))

    def stop(self):
        self.stop_event.set()

    def process_stream(self, source, display_callback=None, status_callback=None):
        self.stop_event.clear()

        # 优化摄像头调用速度（Windows环境下推荐CAP_DSHOW）
        if isinstance(source, int):
            if os.name == 'nt':
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)
            # 降低缓冲，防止实时追踪产生画面延迟堆积
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}。如果是摄像头，请检查ID是否正确或被占用。")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        # 摄像头默认可能读取不到FPS，给一个默认值30
        src_fps = src_fps if src_fps and src_fps > 1e-6 else self.args.fps

        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_folder = osp.join(self.vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        source_name = "camera" if isinstance(source, int) else osp.basename(str(source))
        save_path = osp.join(save_folder, f"{source_name}.mp4")

        vid_writer = None
        if self.args.save_result:
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))

        tracker = BYTETracker(self.args, frame_rate=int(src_fps))
        timer = Timer()
        frame_id = 0
        result_lines = []

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            outputs, img_info = self.predictor.inference(frame, timer)
            object_count = 0
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0], [img_info["height"], img_info["width"]], self.exp.test_size
                )
                online_tlwhs, online_ids, online_scores = [], [], []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / max(1e-6, tlwh[3]) > self.args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        result_lines.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                object_count = len(online_ids)
                timer.toc()
                fps_value = 1.0 / max(1e-5, timer.average_time)
                online_im = plot_tracking(
                    img_info["raw_img"], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps_value
                )
            else:
                timer.toc()
                fps_value = 1.0 / max(1e-5, timer.average_time)
                online_im = img_info["raw_img"]

            if vid_writer is not None:
                vid_writer.write(online_im)

            if status_callback is not None:
                status_callback(TrackStats(fps=fps_value, object_count=object_count, frame_id=frame_id + 1))
            if display_callback is not None and ((frame_id + 1) % DISPLAY_EVERY_N_FRAMES == 0):
                display_callback(online_im)

            frame_id += 1

        cap.release()
        if vid_writer is not None:
            vid_writer.release()
        if self.args.save_result and result_lines:
            res_file = osp.join(save_folder, f"{timestamp}.txt")
            with open(res_file, "w", encoding="utf-8") as f:
                f.writelines(result_lines)


class AppArgs:
    def __init__(self):
        self.exp_file = EXP_FILE
        self.ckpt = CKPT_FILE
        self.device = DEVICE
        self.save_result = SAVE_RESULT_DEFAULT
        self.camid = CAMERA_ID
        self.fps = 30
        self.fp16 = False
        self.fuse = False
        self.trt = False
        self.track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False


class MOTSoftwareApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1300x840")
        self.root.configure(bg="#eef3fb")

        self.user_manager = UserManager()
        self.args = AppArgs()
        self.engine = None
        self.run_thread = None
        self.logged_in_user = None

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.current_user_var = tk.StringVar(value="当前用户：未登录")

        self.camera_id_var = tk.StringVar(value="0")
        self.video_path_var = tk.StringVar()
        self.fps_var = tk.StringVar(value="0.00")
        self.count_var = tk.StringVar(value="0")
        self.frame_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="请先登录")
        self.save_result_var = tk.BooleanVar(value=SAVE_RESULT_DEFAULT)

        self.video_label = None
        self._build_login_view()

    def _clear_root(self):
        for child in self.root.winfo_children():
            child.destroy()

    def _build_login_view(self):
        self._clear_root()
        wrap = tk.Frame(self.root, bg="#eef3fb")
        wrap.pack(fill="both", expand=True)

        card = tk.Frame(wrap, bg="white", bd=1, relief="solid")
        card.place(relx=0.5, rely=0.5, anchor="center", width=430, height=360)

        tk.Label(card, text="多目标跟踪软件登录", bg="white", font=("Microsoft YaHei", 20, "bold")).pack(pady=(30, 25))

        form = tk.Frame(card, bg="white")
        form.pack(fill="x", padx=38)

        tk.Label(form, text="用户名", bg="white", anchor="w", font=("Microsoft YaHei", 11)).pack(fill="x", pady=(0, 6))
        ttk.Entry(form, textvariable=self.username_var, font=("Microsoft YaHei", 11)).pack(fill="x", ipady=6,
                                                                                           pady=(0, 16))

        tk.Label(form, text="密码", bg="white", anchor="w", font=("Microsoft YaHei", 11)).pack(fill="x", pady=(0, 6))
        ttk.Entry(form, textvariable=self.password_var, show="*", font=("Microsoft YaHei", 11)).pack(fill="x", ipady=6,
                                                                                                     pady=(0, 18))

        btns = tk.Frame(card, bg="white")
        btns.pack(pady=10)
        ttk.Button(btns, text="登录", command=self.login).pack(side="left", padx=10, ipadx=12, ipady=6)
        ttk.Button(btns, text="注册", command=self.register).pack(side="left", padx=10, ipadx=12, ipady=6)
        ttk.Button(btns, text="帮助", command=self.show_help).pack(side="left", padx=10, ipadx=12, ipady=6)

    def _build_main_view(self):
        self._clear_root()

        top = tk.Frame(self.root, bg="#dbe8ff", height=56)
        top.pack(fill="x", side="top")
        tk.Button(
            top, text="帮助", command=self.show_help,
            bg="#2f6df6", fg="white", relief="flat",
            font=("Microsoft YaHei", 10, "bold"), padx=16, pady=6
        ).pack(side="left", padx=12, pady=10)
        tk.Label(top, text=WINDOW_TITLE, bg="#dbe8ff", font=("Microsoft YaHei", 16, "bold")).pack(side="left", padx=10)
        tk.Label(top, textvariable=self.current_user_var, bg="#dbe8ff", font=("Microsoft YaHei", 11)).pack(side="right",
                                                                                                           padx=18)

        body = tk.Frame(self.root, bg="#eef3fb")
        body.pack(fill="both", expand=True, padx=12, pady=12)

        left = tk.Frame(body, bg="white", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = tk.Frame(body, bg="white", bd=1, relief="solid", width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.video_label = tk.Label(
            left, text="实时跟踪画面显示区域", bg="#0f172a", fg="white",
            font=("Microsoft YaHei", 16), width=80, height=32
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(right, text="控制面板", bg="white", font=("Microsoft YaHei", 15, "bold")).pack(anchor="w", padx=16,
                                                                                                pady=(18, 10))

        cam_frame = tk.LabelFrame(right, text="设备摄像头", bg="white", font=("Microsoft YaHei", 11, "bold"))
        cam_frame.pack(fill="x", padx=16, pady=10)

        cam_inner = tk.Frame(cam_frame, bg="white")
        cam_inner.pack(fill="x", padx=10, pady=10)

        tk.Label(cam_inner, text="摄像头 ID:", bg="white", font=("Microsoft YaHei", 10)).pack(side="left")
        ttk.Entry(cam_inner, textvariable=self.camera_id_var, width=5).pack(side="left", padx=(5, 10))
        # 当点击此按钮时，将立刻启动追踪
        ttk.Button(cam_inner, text="调用摄像头", command=self.use_camera).pack(side="left")

        file_frame = tk.LabelFrame(right, text="本地文件选择", bg="white", font=("Microsoft YaHei", 11, "bold"))
        file_frame.pack(fill="x", padx=16, pady=10)
        ttk.Entry(file_frame, textvariable=self.video_path_var).pack(fill="x", padx=10, pady=(10, 8), ipady=5)
        ttk.Button(file_frame, text="选择跟踪视频", command=self.choose_video).pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Checkbutton(right, text="保存结果文件", variable=self.save_result_var).pack(anchor="w", padx=18,
                                                                                        pady=(4, 8))

        info_frame = tk.LabelFrame(right, text="运行信息", bg="white", font=("Microsoft YaHei", 11, "bold"))
        info_frame.pack(fill="x", padx=16, pady=10)
        self._info_row(info_frame, "当前FPS", self.fps_var)
        self._info_row(info_frame, "目标数量", self.count_var)
        self._info_row(info_frame, "当前帧号", self.frame_var)
        self._info_row(info_frame, "运行状态", self.status_var)

        btns = tk.Frame(right, bg="white")
        btns.pack(fill="x", padx=16, pady=18)
        ttk.Button(btns, text="开始跟踪", command=self.start_tracking).pack(fill="x", ipady=9, pady=6)
        ttk.Button(btns, text="停止跟踪", command=self.stop_tracking).pack(fill="x", ipady=9, pady=6)
        ttk.Button(btns, text="退出登录", command=self.logout).pack(fill="x", ipady=9, pady=6)

    def _info_row(self, parent, title, value_var):
        row = tk.Frame(parent, bg="white")
        row.pack(fill="x", padx=10, pady=9)
        tk.Label(row, text=title, bg="white", font=("Microsoft YaHei", 10)).pack(side="left")
        tk.Label(row, textvariable=value_var, bg="white", fg="#0b57d0", font=("Microsoft YaHei", 11, "bold")).pack(
            side="right")

    def show_help(self):
        messagebox.showinfo("帮助", HELP_TEXT)

    def register(self):
        ok, msg = self.user_manager.register(self.username_var.get(), self.password_var.get())
        if ok:
            messagebox.showinfo("提示", msg)
        else:
            messagebox.showwarning("提示", msg)

    def login(self):
        ok, msg = self.user_manager.login(self.username_var.get(), self.password_var.get())
        if not ok:
            messagebox.showerror("登录失败", msg)
            return
        self.logged_in_user = self.username_var.get().strip()
        self.current_user_var.set(f"当前用户：{self.logged_in_user}")
        self.status_var.set("已登录，等待选择视频源")
        self._build_main_view()

    def logout(self):
        self.stop_tracking()
        self.logged_in_user = None
        self.username_var.set("")
        self.password_var.set("")
        self.current_user_var.set("当前用户：未登录")
        self.status_var.set("请先登录")
        self._build_login_view()

    # ==========================
    # 核心改动点：一键启动摄像头追踪
    # ==========================
    def use_camera(self):
        cam_id = self.camera_id_var.get().strip()
        if not cam_id.isdigit():
            messagebox.showerror("错误", "摄像头 ID 必须是数字（默认通常为 0）")
            return

        # 将视频源变量赋为数字，代表这是摄像头
        self.video_path_var.set(cam_id)
        # 直接拉起 start_tracking() 函数，实现真正的“点击按钮就调用摄像头”
        self.start_tracking()

    def choose_video(self):
        file_path = filedialog.askopenfilename(
            title="选择待跟踪视频",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.status_var.set("视频已选择，等待开始跟踪")

    def start_tracking(self):
        # 防呆：如果正在运行中，先阻止
        if self.run_thread is not None and self.run_thread.is_alive():
            messagebox.showwarning("提示", "跟踪任务已在运行中，请先点击停止。")
            return

        video_path = self.video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("提示", "请先在“本地文件选择”处选择视频，或点击“调用摄像头”")
            return

        # 判断是摄像头ID还是本地文件路径
        if video_path.isdigit():
            video_source = int(video_path)
        else:
            if not osp.exists(video_path):
                messagebox.showerror("错误", "所选视频文件不存在")
                return
            video_source = video_path

        # 校验固定模型依然存在于磁盘
        if not osp.exists(EXP_FILE):
            messagebox.showerror("错误", f"未找到 exp 文件：{EXP_FILE}")
            return
        if not osp.exists(CKPT_FILE):
            messagebox.showerror("错误", f"未找到权重文件：{CKPT_FILE}")
            return

        self.args.save_result = self.save_result_var.get()
        self.fps_var.set("0.00")
        self.count_var.set("0")
        self.frame_var.set("0")
        self.status_var.set("正在初始化模型并调起摄像头...")

        def worker():
            try:
                self.engine = TrackingEngine(self.args)
                self._set_status_safe("正在进行实时多目标跟踪")
                self.engine.process_stream(
                    source=video_source,
                    display_callback=self.update_video_frame,
                    status_callback=self.update_stats,
                )
                self._set_status_safe("跟踪已结束")
            except Exception as e:
                logger.error(traceback.format_exc())
                self._set_status_safe("运行失败")
                self.root.after(0, lambda: messagebox.showerror("错误", f"程序运行失败：{e}"))

        self.run_thread = threading.Thread(target=worker, daemon=True)
        self.run_thread.start()

    def stop_tracking(self):
        if self.engine is not None:
            self.engine.stop()
        self.status_var.set("已停止")

    def _set_status_safe(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def update_stats(self, stats: TrackStats):
        self.root.after(0, lambda: self.fps_var.set(f"{stats.fps:.2f}"))
        self.root.after(0, lambda: self.count_var.set(str(stats.object_count)))
        self.root.after(0, lambda: self.frame_var.set(str(stats.frame_id)))

    def update_video_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 900, 720
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        if scale < 1.0:
            frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)

        def _update():
            if self.video_label is not None:
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo

        self.root.after(0, _update)


def main():
    root = tk.Tk()
    app = MOTSoftwareApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()