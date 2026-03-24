#!/usr/bin/env python3
import os, time, math, socket, struct
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Device (LEFT hand)
# -----------------------------
GRIPPER_IP = "169.254.186.73"   # DG-5F-M left hand default IP
GRIPPER_PORT = 502              # developer mode TCP port

# -----------------------------
# Model file (must exist locally)
# -----------------------------
MODEL_PATH = "hand_landmarker.task"

# -----------------------------
# Camera resolution presets
# -----------------------------
RESOLUTION_PRESETS = [
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
]
DEFAULT_RESOLUTION_INDEX = 1  # 640x480

# -----------------------------
# Control loop
# -----------------------------
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

# -----------------------------
# Gains / limits
# -----------------------------
KP_FLEX = 0.8                   # motors 4,6,7,8,10,11,12,14,15,16,19,20
KP_SPREAD = 1.2                 # motors 5,9,13 + default spread group
DUTY_LIMIT_FLEX = 200           # motors 4,6,7,8,10,11,12,14,15,16,19,20
DUTY_LIMIT_SPREAD = 250         # motors 5,9,13 + default spread group

KP_THUMB_SPREAD = 1.3           # motors 1,2 = thumb spread
DUTY_LIMIT_THUMB_SPREAD = 300   # motors 1,2 = thumb spread

KP_THUMB_MCP = 1.0              # motor 3 = thumb MCP flex
DUTY_LIMIT_THUMB_MCP = 280      # motor 3 = thumb MCP flex

# Pinky helper
KP_PINKY_SPREAD = 1.5           # motor 17 = pinky spread/base
KP_PINKY_FLEX = 1.1             # motor 18 = pinky proximal flex/base
DUTY_LIMIT_PINKY_SPREAD = 320   # motor 17
DUTY_LIMIT_PINKY_FLEX = 260     # motor 18

# Reduce jitter
DEADBAND_0P1DEG = 8  # 0.8 deg

# -----------------------------
# Motion range mapping
# -----------------------------
# non-thumb flex motors: 6,7,8,10,11,12,14,15,16,18,19,20
FLEX_DEG_DEFAULT = 90.0

# thumb dedicated flex

# motor 3 = thumb MCP flex
FLEX_DEG_THUMB_MCP = 160.0

# motor 4 = thumb IP flex
FLEX_DEG_THUMB_IP  = 130.0

# spread

# default spread motors: 5,9,13,17
SPLAY_GAIN_DEFAULT = 1.0
SPLAY_LIMIT_DEFAULT_DEG = 25.0

# thumb spread motors: 1,2
SPLAY_GAIN_THUMB = 2
SPLAY_LIMIT_THUMB_DEG = 90.0

# smoothing
SMOOTH_ALPHA = 0.35
SPLAY_SMOOTH_ALPHA = 0.35

# -----------------------------
# Fingers & motor mapping
# -----------------------------
FINGER_ORDER = ["finger1", "finger2", "finger3", "finger4", "finger5"]

JOINT_MAP = {
    "finger1": [1, 2, 3, 4],      # thumb: motors 1,2=spread, 3=MCP flex, 4=IP flex
    "finger2": [5, 6, 7, 8],      # index: 5=spread, 6/7/8=flex
    "finger3": [9, 10, 11, 12],   # middle: 9=spread, 10/11/12=flex
    "finger4": [13, 14, 15, 16],  # ring: 13=spread, 14/15/16=flex
    "finger5": [17, 18, 19, 20],  # pinky: 17=spread, 18/19/20=flex
}

# -----------------------------
# Sign maps
# -----------------------------
# TARGET_SIGN:
# "사람 손 동작 의미" -> "모터 목표각 방향" 변환용
TARGET_SIGN = {i: 1 for i in range(1, 21)}

# LEFT hand start guess
for m in [1, 2, 3, 4, 17, 18]:
    TARGET_SIGN[m] = -1

# DUTY_SIGN:
# 목표각 에러 -> 실제 모터 구동 극성 보정용
DUTY_SIGN = {i: 1 for i in range(1, 21)}

# -----------------------------
# Per-motor software enable/disable
# -----------------------------
# 공개 프로토콜에는 개별 모터 통신 OFF 명령이 없어서
# 소프트웨어적으로 target hold + duty=0 처리
MOTOR_ENABLED = {m: True for m in range(1, 21)}

def make_zero_duty():
    return {jid: 0 for jid in range(1, 21)}

def disabled_motor_list():
    return [m for m in range(1, 21) if not MOTOR_ENABLED[m]]

def disabled_motor_text(max_len=60):
    ds = disabled_motor_list()
    if not ds:
        return "None"
    s = ",".join(str(m) for m in ds)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s

def motor_role_name(motor_id):
    role = {
        1: "thumb spread A",
        2: "thumb spread B",
        3: "thumb MCP flex",
        4: "thumb IP flex",
        5: "index spread",
        6: "index flex 1",
        7: "index flex 2",
        8: "index flex 3",
        9: "middle spread",
        10: "middle flex 1",
        11: "middle flex 2",
        12: "middle flex 3",
        13: "ring spread",
        14: "ring flex 1",
        15: "ring flex 2",
        16: "ring flex 3",
        17: "pinky spread",
        18: "pinky flex 1",
        19: "pinky flex 2",
        20: "pinky flex 3",
    }
    return role.get(motor_id, "unknown")

def toggle_motor_enable(motor_id, cur_pos, prev_target, prev_duty):
    """
    개별 모터 제어 on/off 토글.
    off 될 때는 현재 위치 hold, duty 0으로 상태 정리.
    """
    if motor_id < 1 or motor_id > 20:
        return f"Invalid motor id: {motor_id}"

    MOTOR_ENABLED[motor_id] = not MOTOR_ENABLED[motor_id]

    hold = int(cur_pos.get(motor_id, prev_target.get(motor_id, 0)))
    prev_target[motor_id] = hold
    prev_duty[motor_id] = 0

    if not MOTOR_ENABLED[motor_id]:
        return f"Motor {motor_id:02d} ({motor_role_name(motor_id)}) -> OFF"
    else:
        return f"Motor {motor_id:02d} ({motor_role_name(motor_id)}) -> ON"

def enforce_motor_enable_mask(cur_pos, desired=None, target=None, raw=None, duty=None,
                              prev_target=None, prev_duty=None):
    """
    Disabled motor는 현재 위치 hold + duty 0으로 강제
    """
    for m in range(1, 21):
        if MOTOR_ENABLED[m]:
            continue

        hold = int(cur_pos.get(m, 0))

        if desired is not None:
            desired[m] = hold
        if target is not None:
            target[m] = hold
        if raw is not None:
            raw[m] = 0
        if duty is not None:
            duty[m] = 0
        if prev_target is not None:
            prev_target[m] = hold
        if prev_duty is not None:
            prev_duty[m] = 0

def enforce_isolate_mode(cur_pos, active_motor_id, desired=None, target=None, raw=None, duty=None,
                         prev_target=None, prev_duty=None):
    """
    선택한 모터만 살리고 나머지는 모두 hold + duty 0
    """
    if active_motor_id is None:
        return

    for m in range(1, 21):
        if m == active_motor_id:
            continue

        hold = int(cur_pos.get(m, 0))

        if desired is not None:
            desired[m] = hold
        if target is not None:
            target[m] = hold
        if raw is not None:
            raw[m] = 0
        if duty is not None:
            duty[m] = 0
        if prev_target is not None:
            prev_target[m] = hold
        if prev_duty is not None:
            prev_duty[m] = 0

# -----------------------------
# Motor angle limits (deg)
# -----------------------------
MOTOR_LIMITS_DEG = {
    1: (-15, 29),     2: (-77, 22),     3: (-150, 29),    4: (-90, 90),
    5: (-20, 31),     6: (0, 115),      7: (-90, 90),     8: (-90, 90),
    9: (-30, 30),     10: (0, 115),     11: (-90, 90),    12: (-90, 90),
    13: (-32, 15),    14: (0, 110),     15: (-90, 90),    16: (-90, 90),
    17: (-30, 0),     18: (-90, 15),    19: (-90, 90),    20: (-90, 90),
}

def clamp_target_0p1deg(motor_id, target_0p1deg):
    if motor_id not in MOTOR_LIMITS_DEG:
        return int(target_0p1deg)
    lo, hi = MOTOR_LIMITS_DEG[motor_id]
    return int(np.clip(int(target_0p1deg), int(lo * 10), int(hi * 10)))

# -----------------------------
# Target speed limits (deg/s)
# -----------------------------
MAX_SPEED_DEG_S = {m: 100.0 for m in range(1, 21)}
for m in [1, 5, 9, 13, 17]:
    MAX_SPEED_DEG_S[m] = 80.0     # spread motors 1,5,9,13,17
MAX_SPEED_DEG_S[2] = 50.0         # motor 2 = thumb spread
MAX_SPEED_DEG_S[3] = 70.0         # motor 3 = thumb MCP
MAX_SPEED_DEG_S[4] = 90.0         # motor 4 = thumb IP

def rate_limit_target(motor_id, desired_target_0p1deg, prev_target_0p1deg):
    max_deg_s = MAX_SPEED_DEG_S.get(motor_id, 120.0)
    max_delta_0p1deg = int(max_deg_s * 10.0 * DT)
    if max_delta_0p1deg < 1:
        max_delta_0p1deg = 1
    delta = int(desired_target_0p1deg) - int(prev_target_0p1deg)
    if delta > max_delta_0p1deg:
        return int(prev_target_0p1deg) + max_delta_0p1deg
    if delta < -max_delta_0p1deg:
        return int(prev_target_0p1deg) - max_delta_0p1deg
    return int(desired_target_0p1deg)

# -----------------------------
# Step limit vs CURRENT position (deg)
# -----------------------------
MAX_STEP_DEG = {m: 25.0 for m in range(1, 21)}
for m in [1, 5, 9, 13, 17]:
    MAX_STEP_DEG[m] = 10.0        # spread motors 1,5,9,13,17
MAX_STEP_DEG[2] = 10.0            # motor 2 = thumb spread
MAX_STEP_DEG[3] = 15.0            # motor 3 = thumb MCP
MAX_STEP_DEG[4] = 20.0            # motor 4 = thumb IP

def clamp_step_to_current(motor_id, desired_target_0p1deg, current_pos_0p1deg):
    max_step_0p1deg = int(MAX_STEP_DEG.get(motor_id, 25.0) * 10)
    cur = int(current_pos_0p1deg)
    des = int(desired_target_0p1deg)
    if des > cur + max_step_0p1deg:
        return cur + max_step_0p1deg
    if des < cur - max_step_0p1deg:
        return cur - max_step_0p1deg
    return des

# -----------------------------
# Duty slew limit
# -----------------------------
MAX_DUTY_STEP = 40

def slew_limit_duty(motor_id, new_duty, prev_duty):
    prev = int(prev_duty.get(motor_id, 0))
    nd = int(new_duty)
    if nd > prev + MAX_DUTY_STEP:
        nd = prev + MAX_DUTY_STEP
    elif nd < prev - MAX_DUTY_STEP:
        nd = prev - MAX_DUTY_STEP
    prev_duty[motor_id] = nd
    return nd

# -----------------------------
# GLOBAL LOAD LIMITS
# -----------------------------
TOTAL_DUTY_BUDGET = 1700
MAX_ACTIVE_JOINTS = 12
MIN_DUTY_TO_MOVE = 18
PROTECTED_JOINTS = {17, 18}

def apply_global_limits(raw_duty_dict):
    """
    1) 작은 duty 제거
    2) 동시 구동 관절 수 제한 (Top-K) + 보호 관절 유지
    3) 총 듀티 예산 제한 (비율 스케일)
    """
    duty = dict(raw_duty_dict)

    for m in list(duty.keys()):
        if abs(duty[m]) < MIN_DUTY_TO_MOVE:
            duty[m] = 0

    active = [(m, abs(v)) for m, v in duty.items() if v != 0]
    if len(active) > MAX_ACTIVE_JOINTS:
        active.sort(key=lambda x: x[1], reverse=True)

        keep = set(m for m in PROTECTED_JOINTS if duty.get(m, 0) != 0)

        for m, _ in active:
            if len(keep) >= MAX_ACTIVE_JOINTS:
                break
            keep.add(m)

        for m in list(duty.keys()):
            if m not in keep:
                duty[m] = 0

    total = sum(abs(v) for v in duty.values())
    if total > TOTAL_DUTY_BUDGET and total > 0:
        scale = TOTAL_DUTY_BUDGET / total
        for m in list(duty.keys()):
            duty[m] = int(duty[m] * scale)

    return duty

# -----------------------------
# Camera resolution helper
# -----------------------------
def set_camera_resolution(cap, width, height):
    """
    카메라 해상도 설정 및 실제 적용된 해상도 반환
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 실제 설정된 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return actual_width, actual_height

# -----------------------------
# Developer-mode TCP client
# -----------------------------
class DG5FDevClient:
    def __init__(self, ip, port, timeout=0.5):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.ip, self.port))
        print(f"[OK] Connected: {self.ip}:{self.port}")

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf

    def send_only(self, cmd, data=b""):
        length = 2 + 1 + len(data)
        pkt = struct.pack(">H", length) + struct.pack("B", cmd) + data
        self.sock.sendall(pkt)

    def transact(self, cmd, data=b""):
        length = 2 + 1 + len(data)
        pkt = struct.pack(">H", length) + struct.pack("B", cmd) + data
        self.sock.sendall(pkt)
        resp_len = struct.unpack(">H", self._recv_exact(2))[0]
        resp_rest = self._recv_exact(resp_len - 2)
        return resp_rest

    def get_positions(self):
        resp = self.transact(0x01, data=bytes([0x01]))
        if not resp or resp[0] != 0x01:
            raise RuntimeError(f"Unexpected response CMD: {resp[0] if resp else None}")
        payload = resp[1:]
        pos = {}
        i = 0
        while i + 3 <= len(payload):
            jid = payload[i]
            val = struct.unpack(">h", payload[i+1:i+3])[0]  # 0.1deg signed
            pos[jid] = val
            i += 3
        return pos

    def set_duty(self, duty_by_id):
        data = b""
        for jid in range(1, 21):
            duty = int(np.clip(int(duty_by_id.get(jid, 0)), -1000, 1000))
            data += struct.pack("B", jid) + struct.pack(">h", duty)
        self.send_only(0x05, data=data)

# -----------------------------
# Hand tracking
# -----------------------------
FINGER_LANDMARKS = {
    "finger1": [1, 2, 3, 4],      # thumb: landmarks 1,2,3,4 -> motors 1,2,3,4
    "finger2": [5, 6, 7, 8],      # index -> motors 5,6,7,8
    "finger3": [9, 10, 11, 12],   # middle -> motors 9,10,11,12
    "finger4": [13, 14, 15, 16],  # ring -> motors 13,14,15,16
    "finger5": [17, 18, 19, 20],  # pinky -> motors 17,18,19,20
}

def _angle_deg(v1, v2):
    dot = float(np.dot(v1, v2))
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def _curl_from_joint_angle(angle_deg, open_deg=170.0, closed_deg=70.0):
    curl = (open_deg - angle_deg) / (open_deg - closed_deg)
    return float(np.clip(curl, 0.0, 1.0))

def _finger_curl(lms, finger):
    idx = FINGER_LANDMARKS[finger]
    pts = [lms[i] for i in idx]
    if finger == "finger1":
        wrist = lms[0]
        tip = lms[4]
        ref = lms[2]
        cur = np.linalg.norm(tip - wrist)
        maxd = np.linalg.norm(ref - wrist) * 1.4
        if maxd < 1e-9:
            return 0.0
        curl = 1.0 - min(cur / maxd, 1.0)
        return float(np.clip(curl, 0.0, 1.0))
    mcp, pip, dip, tip = pts
    ang_pip = _angle_deg(mcp - pip, dip - pip)
    ang_dip = _angle_deg(pip - dip, tip - dip)
    avg = 0.6 * ang_pip + 0.4 * ang_dip
    return _curl_from_joint_angle(avg)

def _thumb_mcp_ip_curls(lms):
    p1, p2, p3, p4 = lms[1], lms[2], lms[3], lms[4]
    ang_mcp = _angle_deg(p1 - p2, p3 - p2)
    ang_ip  = _angle_deg(p2 - p3, p4 - p3)
    return _curl_from_joint_angle(ang_mcp), _curl_from_joint_angle(ang_ip)

def _draw_landmarks(frame, lms_xy):
    conns = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for a, b in conns:
        cv2.line(frame, lms_xy[a], lms_xy[b], (0,255,0), 2)
    for i, (x, y) in enumerate(lms_xy):
        color, r = (0,0,255), 4
        if i == 0:
            color, r = (255,0,0), 7
        if i in [4,8,12,16,20]:
            color, r = (0,255,255), 6
        cv2.circle(frame, (x, y), r, color, -1)

# -----------------------------
# HUD / Overlay helpers
# -----------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv_safe_text(text: str) -> str:
    if text is None:
        return ""
    rep = {
        "°": "deg",
        "→": "->",
        "←": "<-",
        "✅": "[OK]",
        "⚠️": "[WARN]",
        "－": "-",
        "—": "-",
        "…": "...",
    }
    out = str(text)
    for k, v in rep.items():
        out = out.replace(k, v)
    out = out.encode("ascii", "replace").decode("ascii")
    return out

def put_text_outline(img, text, org, scale=0.5, color=(255,255,255), thickness=1):
    text = cv_safe_text(text)
    cv2.putText(img, text, org, FONT, scale, (0,0,0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

def draw_alpha_box(img, x1, y1, x2, y2, color=(40,40,40), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

def duty_color(v):
    if v > 0:
        return (0, 220, 255)
    elif v < 0:
        return (255, 180, 0)
    return (180, 180, 180)

def joint_label_offset(motor_id):
    local = (motor_id - 1) % 4
    offsets = [
        (-52, -10),  # j0
        (-52,  12),  # j1
        (  8, -10),  # j2
        (  8,  12),  # j3
    ]
    return offsets[local]

def draw_joint_angle_labels(frame, lms_xy, cur_pos, target_pos=None):
    if lms_xy is None or cur_pos is None:
        return

    for motor_id in range(1, 21):
        if motor_id >= len(lms_xy):
            continue
        if motor_id not in cur_pos:
            continue

        x, y = lms_xy[motor_id]
        dx, dy = joint_label_offset(motor_id)

        cur_deg = cur_pos[motor_id] / 10.0
        color = (0, 230, 255) if MOTOR_ENABLED[motor_id] else (120, 120, 120)

        if target_pos is not None and motor_id in target_pos:
            tgt_deg = target_pos[motor_id] / 10.0
            label = f"M{motor_id:02d} C:{cur_deg:+.1f} T:{tgt_deg:+.1f}"
        else:
            label = f"M{motor_id:02d} C:{cur_deg:+.1f}"

        put_text_outline(
            frame,
            label,
            (x + dx, y + dy),
            scale=0.36,
            color=color,
            thickness=1
        )

def draw_duty_panel(frame, duty_dict):
    h, w = frame.shape[:2]
    x1, y1 = w - 320, 10
    x2, y2 = w - 10, 270
    draw_alpha_box(frame, x1, y1, x2, y2, color=(35, 35, 35), alpha=0.62)

    put_text_outline(frame, "Current Duty", (x1 + 10, y1 + 22), scale=0.58, color=(255,255,255), thickness=1)

    total_abs = sum(abs(int(v)) for v in duty_dict.values())
    active = sum(1 for v in duty_dict.values() if int(v) != 0)
    put_text_outline(
        frame,
        f"sum|d|={total_abs} active={active}",
        (x1 + 10, y1 + 44),
        scale=0.45,
        color=(180, 255, 180),
        thickness=1
    )

    col_x = [x1 + 10, x1 + 165]
    start_y = y1 + 70
    row_h = 17

    for idx, motor_id in enumerate(range(1, 21)):
        col = 0 if idx < 10 else 1
        row = idx if idx < 10 else idx - 10
        x = col_x[col]
        y = start_y + row * row_h
        v = int(duty_dict.get(motor_id, 0))
        label = f"M{motor_id:02d}: {v:+4d}"
        if not MOTOR_ENABLED[motor_id]:
            label += " OFF"
        put_text_outline(
            frame,
            label,
            (x, y),
            scale=0.42,
            color=duty_color(v) if MOTOR_ENABLED[motor_id] else (120,120,120),
            thickness=1
        )

def draw_key_panel(frame, emergency_stop, home_zero,
                   motor_input_buffer, flash_text,
                   isolate_mode, isolate_motor_id,
                   resolution_text):
    h, w = frame.shape[:2]
    x1, y2 = 10, h - 10
    x2, y1 = 620, h - 265  # 해상도 정보를 위해 높이 증가
    draw_alpha_box(frame, x1, y1, x2, y2, color=(35,35,35), alpha=0.58)

    put_text_outline(frame, "Keys", (x1 + 10, y1 + 22), scale=0.58, color=(255,255,255), thickness=1)

    isolate_line = "OFF"
    if isolate_mode and isolate_motor_id is not None:
        isolate_line = f"ON  M{isolate_motor_id:02d} ({motor_role_name(isolate_motor_id)})"

    lines = [
        "[Q] Quit",
        "[Z] Zero duty",
        f"[X] EMO   : {'ON' if emergency_stop else 'OFF'}",
        f"[R] HOME0 : {'ON' if home_zero else 'OFF'}",
        "[+/-] Change resolution",
        "[0-9] Type motor id | [Enter] toggle ON/OFF | [Backspace] clear",
        "[T] Isolate selected motor ON/OFF",
        f"Motor input : {motor_input_buffer if motor_input_buffer else '-'}",
        f"Disabled    : {disabled_motor_text()}",
        f"Isolate     : {isolate_line}",
        f"Resolution  : {resolution_text}",
    ]

    y = y1 + 45
    for line in lines:
        put_text_outline(frame, line, (x1 + 10, y), scale=0.42, color=(220,220,220), thickness=1)
        y += 18

    if flash_text:
        put_text_outline(frame, flash_text, (x1 + 10, y + 10), scale=0.48, color=(0,255,255), thickness=1)

def draw_runtime_hud(frame, status_text, lms_xy, cur_pos, target_dict, duty_dict,
                     emergency_stop, home_zero,
                     motor_input_buffer="", flash_text="",
                     isolate_mode=False, isolate_motor_id=None,
                     resolution_text=""):
    status_color = (0,255,0)
    if emergency_stop:
        status_color = (0,0,255)
    elif home_zero:
        status_color = (0,165,255)

    low = status_text.lower()
    if "error" in low or "show your hand" in low or "emergency" in low:
        status_color = (0, 0, 255)
    elif "home0" in low:
        status_color = (0, 165, 255)

    put_text_outline(frame, status_text, (10, 30), scale=0.65, color=status_color, thickness=2)

    draw_joint_angle_labels(frame, lms_xy, cur_pos, target_dict)
    draw_duty_panel(frame, duty_dict)
    draw_key_panel(
        frame,
        emergency_stop,
        home_zero,
        motor_input_buffer,
        flash_text,
        isolate_mode,
        isolate_motor_id,
        resolution_text
    )

# -----------------------------
# splay helpers
# -----------------------------
def _unit2(v):
    n = float(np.linalg.norm(v))
    return v / (n + 1e-9)

def _signed_angle_2d(a, b):
    a = _unit2(a)
    b = _unit2(b)
    return math.degrees(math.atan2(a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1]))

def _finger_dir_2d(lms_np, mcp_idx, tip_idx):
    v = lms_np[tip_idx] - lms_np[mcp_idx]
    return np.array([v[0], v[1]], dtype=np.float32)

def compute_splay_deg(lms_np):
    dirs = {
        "finger2": _finger_dir_2d(lms_np, 5, 8),
        "finger3": _finger_dir_2d(lms_np, 9, 12),
        "finger4": _finger_dir_2d(lms_np, 13, 16),
        "finger5": _finger_dir_2d(lms_np, 17, 20),
        "finger1": _finger_dir_2d(lms_np, 2, 4),
    }
    base = dirs["finger3"]
    return {
        "finger3": 0.0,
        "finger2": _signed_angle_2d(base, dirs["finger2"]),
        "finger4": _signed_angle_2d(base, dirs["finger4"]),
        "finger5": _signed_angle_2d(base, dirs["finger5"]),
        "finger1": _signed_angle_2d(base, dirs["finger1"]),
    }

class HandTrackerTasks:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("[OK] Hand tracker ready")

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)

        if not res.hand_landmarks:
            return frame_bgr, None, None, None, None

        # -----------------------------
        # LEFT hand only
        # 현재 main()에서 frame을 flip(frame, 1) 한 뒤 여기로 들어오므로
        # MediaPipe handedness의 "Left"를 실제 왼손으로 사용
        # -----------------------------
        hand_idx = None
        if hasattr(res, "handedness") and res.handedness:
            for idx, handed_list in enumerate(res.handedness):
                if handed_list and len(handed_list) > 0:
                    label = handed_list[0].category_name
                    if label == "Right":
                        hand_idx = idx
                        break

        if hand_idx is None:
            # 왼손이 아니면 무시
            return frame_bgr, None, None, None, None

        lms = res.hand_landmarks[hand_idx]
        lms_np = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        lms_xy = [(int(p.x*w), int(p.y*h)) for p in lms]
        _draw_landmarks(frame_bgr, lms_xy)

        curls = {f: _finger_curl(lms_np, f) for f in FINGER_ORDER}
        splay = compute_splay_deg(lms_np)
        thumb_mcp_curl, thumb_ip_curl = _thumb_mcp_ip_curls(lms_np)
        return frame_bgr, curls, splay, (thumb_mcp_curl, thumb_ip_curl), lms_xy

# -----------------------------
# Control helpers
# -----------------------------
def to_duty(err_0p1deg, kp, lim, motor_id):
    if abs(err_0p1deg) < DEADBAND_0P1DEG:
        return 0
    d = int(kp * err_0p1deg)
    d = int(np.clip(d, -lim, lim))
    return DUTY_SIGN[motor_id] * d

def curl_to_flex_deg(curl_now, flex_deg):
    c = float(np.clip(curl_now, 0.0, 1.0))
    return c * flex_deg

# -----------------------------
# Main
# -----------------------------
def main():
    print("===================================================")
    print("DG-5F-M DevMode Teleop")
    print("keys: q quit | z zero | x EMO | r HOME0")
    print("      +/- resolution | number keys + Enter: motor ON/OFF toggle")
    print("      number keys + T: isolate selected motor ON/OFF")
    print("===================================================")

    gr = DG5FDevClient(GRIPPER_IP, GRIPPER_PORT, timeout=0.5)
    gr.connect()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        gr.close()
        raise RuntimeError("Camera open failed")

    # 초기 해상도 설정
    resolution_index = DEFAULT_RESOLUTION_INDEX
    current_width, current_height = set_camera_resolution(
        cap, 
        RESOLUTION_PRESETS[resolution_index][0],
        RESOLUTION_PRESETS[resolution_index][1]
    )
    print(f"[OK] Initial resolution: {current_width}x{current_height}")

    tracker = HandTrackerTasks()

    smooth_curl = {f: 0.0 for f in FINGER_ORDER}
    smooth_splay = {f: 0.0 for f in FINGER_ORDER}
    smooth_thumb_mcp = 0.0
    smooth_thumb_ip = 0.0

    emergency_stop = False
    home_zero = False

    prev_target = {m: 0 for m in range(1, 21)}
    prev_target_valid = False

    prev_duty = {m: 0 for m in range(1, 21)}
    last_sent_duty = {m: 0 for m in range(1, 21)}
    last_target = {m: 0 for m in range(1, 21)}  # HUD 표시용 target(0.1deg)

    motor_input_buffer = ""
    flash_text = ""
    flash_t_end = 0.0

    # isolate mode state
    isolate_mode = False
    isolate_motor_id = None

    last = time.time()

    def reset_targets_to_current(cur_pos):
        nonlocal prev_target, prev_target_valid, last_target
        for m in range(1, 21):
            v = int(cur_pos.get(m, 0))
            prev_target[m] = v
            last_target[m] = v
        prev_target_valid = True

    def reset_duty_state():
        for m in range(1, 21):
            prev_duty[m] = 0

    def enter_isolate_mode(motor_id, cur_pos):
        nonlocal isolate_mode, isolate_motor_id
        if motor_id < 1 or motor_id > 20:
            return f"Invalid motor id for isolate: {motor_id}"
        isolate_mode = True
        isolate_motor_id = motor_id
        for m in range(1, 21):
            prev_target[m] = int(cur_pos.get(m, 0))
            prev_duty[m] = 0
        return f"Isolate -> ON for M{motor_id:02d} ({motor_role_name(motor_id)})"

    def exit_isolate_mode(cur_pos):
        nonlocal isolate_mode, isolate_motor_id
        for m in range(1, 21):
            prev_target[m] = int(cur_pos.get(m, 0))
            prev_duty[m] = 0
        isolate_mode = False
        isolate_motor_id = None
        return "Isolate -> OFF"

    def change_resolution(direction):
        """
        해상도 변경 함수 (+1: 증가, -1: 감소)
        """
        nonlocal resolution_index, current_width, current_height, flash_text, flash_t_end
        
        new_index = resolution_index + direction
        
        if new_index < 0:
            new_index = 0
            flash_text = "Already at minimum resolution"
            flash_t_end = time.time() + 1.5
            return
        
        if new_index >= len(RESOLUTION_PRESETS):
            new_index = len(RESOLUTION_PRESETS) - 1
            flash_text = "Already at maximum resolution"
            flash_t_end = time.time() + 1.5
            return
        
        resolution_index = new_index
        target_width, target_height = RESOLUTION_PRESETS[resolution_index]
        
        current_width, current_height = set_camera_resolution(cap, target_width, target_height)
        
        flash_text = f"Resolution changed to {current_width}x{current_height}"
        flash_t_end = time.time() + 2.0
        print(f"[OK] Resolution changed to {current_width}x{current_height}")

    try:
        while True:
            now = time.time()
            if now - last < DT:
                time.sleep(max(0.0, DT - (now - last)))
            last = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            frame, curls, splay, thumb_pair, lms_xy = tracker.process(frame)

            key = cv2.waitKey(1) & 0xFF

            if time.time() > flash_t_end:
                flash_text = ""

            if key == ord("q"):
                break

            # 해상도 변경 키 처리
            if key == ord("+") or key == ord("="):
                change_resolution(1)
            elif key == ord("-") or key == ord("_"):
                change_resolution(-1)

            # number input
            if ord('0') <= key <= ord('9'):
                if len(motor_input_buffer) < 2:
                    motor_input_buffer += chr(key)

            if key in (8, 127):  # backspace
                motor_input_buffer = motor_input_buffer[:-1]

            # get current positions first
            try:
                cur = gr.get_positions()
            except Exception as e:
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                reset_duty_state()
                last_sent_duty = make_zero_duty()

                resolution_text = f"{current_width}x{current_height}"
                draw_runtime_hud(
                    frame,
                    f"Comm error: {e}",
                    lms_xy,
                    None,
                    None,
                    last_sent_duty,
                    emergency_stop,
                    home_zero,
                    motor_input_buffer,
                    flash_text,
                    isolate_mode,
                    isolate_motor_id,
                    resolution_text
                )
                cv2.imshow("DG-5F Dev Teleop", frame)
                prev_target_valid = False
                continue

            # Enter: toggle motor enable
            if key in (10, 13):
                if motor_input_buffer:
                    motor_id = int(motor_input_buffer)
                    msg = toggle_motor_enable(motor_id, cur, prev_target, prev_duty)
                    flash_text = msg
                    flash_t_end = time.time() + 1.5
                    motor_input_buffer = ""

            # T: isolate selected motor on/off
            if key == ord("t"):
                if isolate_mode:
                    flash_text = exit_isolate_mode(cur)
                    flash_t_end = time.time() + 1.5
                else:
                    if motor_input_buffer:
                        motor_id = int(motor_input_buffer)
                        flash_text = enter_isolate_mode(motor_id, cur)
                        flash_t_end = time.time() + 1.5
                        motor_input_buffer = ""

            if key == ord("z"):
                gr.set_duty(make_zero_duty())
                reset_duty_state()
                last_sent_duty = make_zero_duty()

            if key == ord("x"):
                emergency_stop = not emergency_stop
                if emergency_stop:
                    home_zero = False
                print(f"EMO={emergency_stop}")

            if key == ord("r"):
                home_zero = not home_zero
                if home_zero:
                    emergency_stop = False
                print(f"HOME0={home_zero}")

            if not prev_target_valid:
                reset_targets_to_current(cur)
                reset_duty_state()

            resolution_text = f"{current_width}x{current_height}"

            # EMO
            if emergency_stop:
                gr.set_duty(make_zero_duty())
                reset_targets_to_current(cur)
                reset_duty_state()
                last_sent_duty = make_zero_duty()

                draw_runtime_hud(
                    frame,
                    "EMERGENCY STOP",
                    lms_xy,
                    cur,
                    last_target,
                    last_sent_duty,
                    emergency_stop,
                    home_zero,
                    motor_input_buffer,
                    flash_text,
                    isolate_mode,
                    isolate_motor_id,
                    resolution_text
                )

                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # HOME0
            if home_zero:
                desired = {m: clamp_target_0p1deg(m, 0) for m in range(1, 21)}
                for m in range(1, 21):
                    desired[m] = clamp_step_to_current(m, desired[m], cur.get(m, 0))

                enforce_motor_enable_mask(
                    cur,
                    desired=desired,
                    prev_target=prev_target,
                    prev_duty=prev_duty
                )

                if isolate_mode and isolate_motor_id is not None:
                    enforce_isolate_mode(
                        cur,
                        isolate_motor_id,
                        desired=desired,
                        prev_target=prev_target,
                        prev_duty=prev_duty
                    )

                target = {}
                for m in range(1, 21):
                    target[m] = rate_limit_target(m, desired[m], prev_target[m])
                    prev_target[m] = target[m]

                last_target = dict(target)

                enforce_motor_enable_mask(
                    cur,
                    target=target,
                    prev_target=prev_target,
                    prev_duty=prev_duty
                )

                if isolate_mode and isolate_motor_id is not None:
                    enforce_isolate_mode(
                        cur,
                        isolate_motor_id,
                        target=target,
                        prev_target=prev_target,
                        prev_duty=prev_duty
                    )

                raw = {}
                for m in range(1, 21):
                    err = target[m] - cur.get(m, 0)

                    if m in [1, 2]:
                        kp, lim = KP_THUMB_SPREAD, DUTY_LIMIT_THUMB_SPREAD
                    elif m == 3:
                        kp, lim = KP_THUMB_MCP, DUTY_LIMIT_THUMB_MCP
                    elif m == 17:
                        kp, lim = KP_PINKY_SPREAD, DUTY_LIMIT_PINKY_SPREAD
                    elif m == 18:
                        kp, lim = KP_PINKY_FLEX, DUTY_LIMIT_PINKY_FLEX
                    elif m in [5, 9, 13]:
                        kp, lim = KP_SPREAD, DUTY_LIMIT_SPREAD
                    else:
                        kp, lim = KP_FLEX, DUTY_LIMIT_FLEX

                    raw[m] = to_duty(err, kp, lim, m)

                enforce_motor_enable_mask(cur, raw=raw)
                if isolate_mode and isolate_motor_id is not None:
                    enforce_isolate_mode(cur, isolate_motor_id, raw=raw)

                raw = apply_global_limits(raw)

                enforce_motor_enable_mask(cur, raw=raw)
                if isolate_mode and isolate_motor_id is not None:
                    enforce_isolate_mode(cur, isolate_motor_id, raw=raw)

                duty = {}
                for m in range(1, 21):
                    duty[m] = slew_limit_duty(m, raw[m], prev_duty)

                enforce_motor_enable_mask(cur, duty=duty, prev_duty=prev_duty)
                if isolate_mode and isolate_motor_id is not None:
                    enforce_isolate_mode(cur, isolate_motor_id, duty=duty, prev_duty=prev_duty)

                gr.set_duty(duty)
                last_sent_duty = dict(duty)

                status = "HOME0 (global+step+speed+slew)"
                if isolate_mode and isolate_motor_id is not None:
                    status += f" | ISOLATE M{isolate_motor_id:02d}"

                draw_runtime_hud(
                    frame,
                    status,
                    lms_xy,
                    cur,
                    last_target,
                    last_sent_duty,
                    emergency_stop,
                    home_zero,
                    motor_input_buffer,
                    flash_text,
                    isolate_mode,
                    isolate_motor_id,
                    resolution_text
                )

                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # No hand -> stop and sync targets
            if curls is None or splay is None or thumb_pair is None:
                gr.set_duty(make_zero_duty())
                reset_targets_to_current(cur)
                reset_duty_state()
                last_sent_duty = make_zero_duty()

                draw_runtime_hud(
                    frame,
                    "Show your hand",
                    lms_xy,
                    cur,
                    last_target,
                    last_sent_duty,
                    emergency_stop,
                    home_zero,
                    motor_input_buffer,
                    flash_text,
                    isolate_mode,
                    isolate_motor_id,
                    resolution_text
                )

                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # smoothing
            for f in FINGER_ORDER:
                smooth_curl[f] = (1.0 - SMOOTH_ALPHA) * smooth_curl[f] + SMOOTH_ALPHA * curls[f]
                curls[f] = smooth_curl[f]
                smooth_splay[f] = (1.0 - SPLAY_SMOOTH_ALPHA) * smooth_splay[f] + SPLAY_SMOOTH_ALPHA * splay[f]
                splay[f] = smooth_splay[f]

            thumb_mcp_curl, thumb_ip_curl = thumb_pair
            smooth_thumb_mcp = (1.0 - SMOOTH_ALPHA) * smooth_thumb_mcp + SMOOTH_ALPHA * thumb_mcp_curl
            smooth_thumb_ip  = (1.0 - SMOOTH_ALPHA) * smooth_thumb_ip  + SMOOTH_ALPHA * thumb_ip_curl
            thumb_mcp_curl, thumb_ip_curl = smooth_thumb_mcp, smooth_thumb_ip

            # 1) desired targets (0.1deg)
            desired = {m: prev_target[m] for m in range(1, 21)}  # default hold

            for f in FINGER_ORDER:
                j0, j1, j2, j3 = JOINT_MAP[f]

                # spread desired
                if f == "finger1":
                    spread_deg = float(np.clip(
                        SPLAY_GAIN_THUMB * splay[f],
                        -SPLAY_LIMIT_THUMB_DEG,
                        SPLAY_LIMIT_THUMB_DEG
                    ))  # thumb spread -> motors 1,2
                else:
                    spread_deg = float(np.clip(
                        SPLAY_GAIN_DEFAULT * splay[f],
                        -SPLAY_LIMIT_DEFAULT_DEG,
                        SPLAY_LIMIT_DEFAULT_DEG
                    ))  # non-thumb spread -> motors 5,9,13,17

                spread_cmd_0p1 = int(spread_deg * 10)

                if f == "finger1":
                    # thumb uses dedicated MCP/IP curls
                    # motor 3 = thumb MCP
                    mcp_deg = curl_to_flex_deg(
                        thumb_mcp_curl,
                        FLEX_DEG_THUMB_MCP
                    )
                    # motor 4 = thumb IP
                    ip_deg = curl_to_flex_deg(
                        thumb_ip_curl,
                        FLEX_DEG_THUMB_IP
                    )

                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * spread_cmd_0p1)      # motor 1
                    desired[j1] = clamp_target_0p1deg(j1, TARGET_SIGN[j1] * spread_cmd_0p1)      # motor 2
                    desired[j2] = clamp_target_0p1deg(j2, TARGET_SIGN[j2] * int(mcp_deg * 10))   # motor 3
                    desired[j3] = clamp_target_0p1deg(j3, TARGET_SIGN[j3] * int(ip_deg * 10))    # motor 4

                else:
                    # non-thumb flex -> motors 6/7/8, 10/11/12, 14/15/16, 18/19/20
                    flex_deg = curl_to_flex_deg(
                        curls[f],
                        FLEX_DEG_DEFAULT
                    )
                    flex_0p1 = int(flex_deg * 10)

                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * spread_cmd_0p1)  # spread motors 5,9,13,17
                    desired[j1] = clamp_target_0p1deg(j1, TARGET_SIGN[j1] * flex_0p1)        # flex motors 6,10,14,18
                    desired[j2] = clamp_target_0p1deg(j2, TARGET_SIGN[j2] * flex_0p1)        # flex motors 7,11,15,19
                    desired[j3] = clamp_target_0p1deg(j3, TARGET_SIGN[j3] * flex_0p1)        # flex motors 8,12,16,20

            # 2) step limit vs current
            for m in range(1, 21):
                desired[m] = clamp_step_to_current(m, desired[m], cur.get(m, 0))

            enforce_motor_enable_mask(
                cur,
                desired=desired,
                prev_target=prev_target,
                prev_duty=prev_duty
            )

            if isolate_mode and isolate_motor_id is not None:
                enforce_isolate_mode(
                    cur,
                    isolate_motor_id,
                    desired=desired,
                    prev_target=prev_target,
                    prev_duty=prev_duty
                )

            # 3) speed limit vs prev_target
            target = {}
            for m in range(1, 21):
                target[m] = rate_limit_target(m, desired[m], prev_target[m])
                prev_target[m] = target[m]

            last_target = dict(target)
            enforce_motor_enable_mask(
                cur,
                target=target,
                prev_target=prev_target,
                prev_duty=prev_duty
            )

            if isolate_mode and isolate_motor_id is not None:
                enforce_isolate_mode(
                    cur,
                    isolate_motor_id,
                    target=target,
                    prev_target=prev_target,
                    prev_duty=prev_duty
                )

            # 4) P-control -> raw duty
            raw = {}
            for f in FINGER_ORDER:
                j0, j1, j2, j3 = JOINT_MAP[f]

                # spread
                e0 = target[j0] - cur.get(j0, 0)

                if f == "finger1":
                    raw[j0] = to_duty(e0, KP_THUMB_SPREAD, DUTY_LIMIT_THUMB_SPREAD, j0)  # motor 1
                elif f == "finger5":
                    raw[j0] = to_duty(e0, KP_PINKY_SPREAD, DUTY_LIMIT_PINKY_SPREAD, j0)  # motor 17
                else:
                    raw[j0] = to_duty(e0, KP_SPREAD, DUTY_LIMIT_SPREAD, j0)               # motors 5,9,13

                if f == "finger1":
                    e1 = target[j1] - cur.get(j1, 0)
                    raw[j1] = to_duty(e1, KP_THUMB_SPREAD, DUTY_LIMIT_THUMB_SPREAD, j1)   # motor 2

                    e2 = target[j2] - cur.get(j2, 0)
                    raw[j2] = to_duty(e2, KP_THUMB_MCP, DUTY_LIMIT_THUMB_MCP, j2)         # motor 3

                    e3 = target[j3] - cur.get(j3, 0)
                    raw[j3] = to_duty(e3, KP_FLEX, DUTY_LIMIT_FLEX, j3)                   # motor 4

                elif f == "finger5":
                    e1 = target[j1] - cur.get(j1, 0)
                    e2 = target[j2] - cur.get(j2, 0)
                    e3 = target[j3] - cur.get(j3, 0)

                    raw[j1] = to_duty(e1, KP_PINKY_FLEX, DUTY_LIMIT_PINKY_FLEX, j1)       # motor 18
                    raw[j2] = to_duty(e2, KP_FLEX, DUTY_LIMIT_FLEX, j2)                   # motor 19
                    raw[j3] = to_duty(e3, KP_FLEX, DUTY_LIMIT_FLEX, j3)                   # motor 20

                else:
                    e1 = target[j1] - cur.get(j1, 0)
                    e2 = target[j2] - cur.get(j2, 0)
                    e3 = target[j3] - cur.get(j3, 0)
                    raw[j1] = to_duty(e1, KP_FLEX, DUTY_LIMIT_FLEX, j1)
                    raw[j2] = to_duty(e2, KP_FLEX, DUTY_LIMIT_FLEX, j2)
                    raw[j3] = to_duty(e3, KP_FLEX, DUTY_LIMIT_FLEX, j3)

            enforce_motor_enable_mask(cur, raw=raw)

            if isolate_mode and isolate_motor_id is not None:
                enforce_isolate_mode(cur, isolate_motor_id, raw=raw)

            raw = apply_global_limits(raw)

            enforce_motor_enable_mask(cur, raw=raw)

            if isolate_mode and isolate_motor_id is not None:
                enforce_isolate_mode(cur, isolate_motor_id, raw=raw)

            # 6) duty slew (final duty)
            duty = {}
            for m in range(1, 21):
                duty[m] = slew_limit_duty(m, raw.get(m, 0), prev_duty)

            enforce_motor_enable_mask(cur, duty=duty, prev_duty=prev_duty)

            if isolate_mode and isolate_motor_id is not None:
                enforce_isolate_mode(cur, isolate_motor_id, duty=duty, prev_duty=prev_duty)

            write_error_text = None
            try:
                gr.set_duty(duty)
                last_sent_duty = dict(duty)
            except Exception as e:
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                reset_targets_to_current(cur)
                reset_duty_state()
                last_sent_duty = make_zero_duty()
                write_error_text = f"Write error: {e}"

            status = f"TELEOP | Budget={TOTAL_DUTY_BUDGET} TopK={MAX_ACTIVE_JOINTS}"
            if isolate_mode and isolate_motor_id is not None:
                status += f" | ISOLATE M{isolate_motor_id:02d} ({motor_role_name(isolate_motor_id)})"
            if write_error_text is not None:
                status = write_error_text

            draw_runtime_hud(
                frame,
                status,
                lms_xy,
                cur,
                last_target,
                last_sent_duty,
                emergency_stop,
                home_zero,
                motor_input_buffer,
                flash_text,
                isolate_mode,
                isolate_motor_id,
                resolution_text
            )

            cv2.imshow("DG-5F Dev Teleop", frame)

    finally:
        try:
            gr.set_duty(make_zero_duty())
        except:
            pass
        gr.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Exit cleanly")

if __name__ == "__main__":
    main()
