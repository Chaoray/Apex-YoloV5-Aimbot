import dxcam
import win32api, win32con, win32gui
import tkinter as tk
import torch
import time
import threading
import math

window_handle = win32gui.FindWindow(None, "Apex Legends")
if window_handle == None:
    print("Apex window not found")
    exit(1)

WINDOW_RECT = win32gui.GetWindowRect(window_handle)
WINDOW_SIZE = (WINDOW_RECT[2] - WINDOW_RECT[0], WINDOW_RECT[3] - WINDOW_RECT[1])

# CONSTANTS
CAPTURE_SIZE = 320
HEAD = 0.7
AIM_THRESHOLD = 0
MOVE_SPEED = 0.66
CONF = 0.35

ONE_DEGREE_MOVEPX_X = 5400 / 360
ONE_DEGREE_MOVEPX_Y = 2670 / 180
FPS = 75
FOV = 110
SENSITIVE = 3.0


CAMERA_TO_SCREEN = int(abs(WINDOW_SIZE[0] / 2 / math.tan(FOV / 2)))
CENTER = (CAPTURE_SIZE // 2, CAPTURE_SIZE // 2)


# control vars
head_var = None
move_speed_var = None
aim_threshold_var = None
conf_var = None

model = torch.hub.load(
    "./yolov5",
    "custom",
    path="./weights/best.pt",
    source="local",
    _verbose=False,
)
model.cuda()

camera = dxcam.create(output_idx=0)

running = False


def getNearest(points, current):
    min_dist = 10000000000
    target = points[0]
    for i in range(len(points)):
        dist = (points[i][0] - current[0]) ** 2 + (points[i][1] - current[1]) ** 2
        if dist < min_dist:
            min_dist = dist
            target = points[i]
    return target


def calcMouseMovePx(dx, dy):
    angle_x = math.atan(dx / CAMERA_TO_SCREEN)
    angle_y = math.atan(dy / CAMERA_TO_SCREEN)
    return angle_x * ONE_DEGREE_MOVEPX_X, angle_y * ONE_DEGREE_MOVEPX_Y


def detect():
    capture_region = (
        (WINDOW_RECT[0] + WINDOW_RECT[2]) // 2 - CAPTURE_SIZE // 2,
        (WINDOW_RECT[1] + WINDOW_RECT[3]) // 2 - CAPTURE_SIZE // 2,
        (WINDOW_RECT[0] + WINDOW_RECT[2]) // 2 + CAPTURE_SIZE // 2,
        (WINDOW_RECT[1] + WINDOW_RECT[3]) // 2 + CAPTURE_SIZE // 2,
    )
    left, top = capture_region[0], capture_region[1]

    camera.start(target_fps=FPS, region=capture_region)

    while True:
        if not running:
            break

        frame = camera.get_latest_frame()
        if not win32api.GetKeyState(0x02) < 0:  # right button down
            time.sleep(0.001)
            continue

        flags, _, _ = win32gui.GetCursorInfo()
        if flags:
            time.sleep(0.001)
            continue

        detections = model(frame, size=CAPTURE_SIZE)

        if not detections.xyxy[0].shape[0]:
            continue

        detections = detections.xyxy[0].cpu().numpy().astype(int)
        targets = []
        for i in range(len(detections)):
            if (detections[i][2] - detections[i][0]) / CAPTURE_SIZE < aim_threshold_var.get():
                continue
            targets.append([
                (detections[i][0] + detections[i][2]) >> 1,
                int((detections[i][1] * head_var.get() + detections[i][3] * (1 - head_var.get())))
            ])
        
        if not len(targets):
            continue

        target = getNearest(targets, CENTER)

        (move_x, move_y) = calcMouseMovePx(target[0] - CENTER[0], target[1] - CENTER[1])

        interp = (
            move_x * move_speed_var.get(),
            move_y * move_speed_var.get(),
        )

        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(interp[0]), int(interp[1]))


model_thread = None


def start():
    global running
    global state
    if running:
        return
    running = True
    state.config(text="Started")

    global model_thread
    model_thread = threading.Thread(target=detect)
    model_thread.daemon = True
    model_thread.start()


def stop():
    global running
    global state
    running = False
    state.config(text="Stopped")


def conf_change(e):
    global conf_var
    model.conf = conf_var.get()


def controls():
    global move_speed_var
    global head_var
    global aim_threshold_var
    global conf_var

    tk.Label(text="[ Apex Yolov5 AI Predict ]", font=("Arial", 15)).pack(side="top")

    tk.Label(text="Head Ratio").place(x=10, y=50)
    head = tk.Scale(
        from_=0.0,
        to_=1.0,
        length=200,
        resolution=0.01,
        orient="horizontal",
        variable=head_var,
    )
    head.place(x=100, y=35)

    tk.Label(text="Move Speed").place(x=10, y=90)
    move_speed = tk.Scale(
        from_=0.0,
        to_=3.0,
        length=200,
        resolution=0.01,
        orient="horizontal",
        variable=move_speed_var,
    )
    move_speed.place(x=100, y=75)

    tk.Label(text="Aim Threshold").place(x=10, y=130)
    aim_threshold = tk.Scale(
        from_=0.0,
        to_=1.0,
        length=200,
        resolution=0.01,
        orient="horizontal",
        variable=aim_threshold_var,
    )
    aim_threshold.place(x=100, y=115)

    tk.Label(text="Model Conf").place(x=10, y=170)
    model_conf = tk.Scale(
        from_=0.0,
        to_=1.0,
        length=200,
        resolution=0.01,
        orient="horizontal",
        variable=conf_var,
        command=conf_change,
    )
    model_conf.place(x=100, y=155)

    global state
    state = tk.Label(text="Stopped")
    state.place(x=10, y=210)

    start_button = tk.Button(text="Start", command=start)
    start_button.place(x=100, y=210)

    stop_button = tk.Button(text="Stop", command=stop)
    stop_button.place(x=200, y=210)


root = None


def on_closing():
    global model
    global camera
    global running
    running = False
    del model
    del camera
    root.destroy()


def main():
    global root
    root = createWindow()
    root.mainloop()


def createWindow():
    global head_var
    global move_speed_var
    global aim_threshold_var
    global conf_var

    window = tk.Tk()
    window.wm_protocol("WM_DELETE_WINDOW", on_closing)

    head_var = tk.DoubleVar(value=HEAD)
    move_speed_var = tk.DoubleVar(value=MOVE_SPEED)
    aim_threshold_var = tk.DoubleVar(value=AIM_THRESHOLD)
    conf_var = tk.DoubleVar(value=CONF)

    window.title("Apex AI")
    window.geometry("350x400")
    window.resizable(False, False)
    controls()
    return window


if __name__ == "__main__":
    main()
