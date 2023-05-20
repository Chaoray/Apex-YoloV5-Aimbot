import torch
import dxcam
import win32api, win32con, win32gui
import signal
import sys

# CONSTANTS
SCREEN_SIZE = (1920, 1080)
DETECT_SIZE = (640, 360)
HEAD = 0.7
REVERSE_HEAD = 1 - HEAD
PREAIM = 1
MOVE_SPEED = 0.7
FPS = 75
CONF = 0.35

model = torch.hub.load(
    "./yolov5",
    "custom",
    path="./weights/best.pt",
    source="local",
    _verbose=False,
)

model.cuda()
model.conf = CONF

left, top = (SCREEN_SIZE[0] - DETECT_SIZE[0]) // 2, (
    SCREEN_SIZE[1] - DETECT_SIZE[1]
) // 2
right, bottom = (SCREEN_SIZE[0] + DETECT_SIZE[0]) // 2, (
    SCREEN_SIZE[1] + DETECT_SIZE[1]
) // 2

camera = dxcam.create(output_idx=0)
camera.start(target_fps=FPS, region=(left, top, right, bottom))

def signal_handler(signum, frame):
    global camera
    if signum == signal.SIGINT.value:
        camera.stop()
        del camera
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def getNearest(points, current):
    min_dist = 1000000
    index = 0
    for i in range(len(points)):
        dist = abs(points[i][0] - current[0]) + abs(points[i][1] - current[1])
        if dist < min_dist:
            min_dist = dist
            index = i
    return index

print('Done.')

last_vel = (0, 0)
last_target = (0, 0)
first_target = True

while True:
    if not win32api.GetKeyState(0x02) < 0:
        first_target = True
        continue

    flags, _, _ = win32gui.GetCursorInfo()
    if flags:
        continue

    frame = camera.get_latest_frame()
    detections = model(frame)

    if not detections.xyxy[0].shape[0]:
        last_vel = (0, 0)
        continue

    detections = detections.xyxy[0].cpu().numpy().astype(int)
    detections = [
        [
            (detection[0] + detection[2]) >> 1,
            int((detection[1] * HEAD + detection[3] * REVERSE_HEAD)),
        ]
        for detection in detections
    ]

    (px, py) = win32api.GetCursorPos()

    target = detections[getNearest(detections, (px - left, py - top))]
    
    if not first_target:
        last_vel = (target[0] - last_target[0], target[1] - last_target[1])
    else:
        first_target = False
    
    pre_target = (
        left + target[0] + last_vel[0] * PREAIM,
        top + target[1] + last_vel[1] * PREAIM
    )
    
    interp = (
        (pre_target[0] - px) * MOVE_SPEED,
        (pre_target[1] - py) * MOVE_SPEED
    )

    last_target = target

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(interp[0]), int(interp[1]))

