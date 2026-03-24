#!/usr/bin/env python3
import os, time, math, socket, struct
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Configuration
# -----------------------------
GRIPPER_IP = "169.254.186.73"
GRIPPER_PORT = 502
MODEL_PATH = "hand_landmarker.task"
RESOLUTION_PRESETS = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]
DEFAULT_RESOLUTION_INDEX = 1

# Control parameters
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ

# Motor configuration (KP, DUTY_LIMIT, ROLE_NAME)
MOTOR_CONFIG = {
    1: (1.3, 300, "thumb opposition"),  # Rotational spread
    2: (1.2, 320, "thumb CMC flex"),  # Base flexion/extension (FIXED!)
    3: (1.0, 280, "thumb MCP flex"),  # Middle joint
    4: (0.8, 200, "thumb IP flex"),  # Tip joint
    5: (1.2, 250, "index spread"), 6: (0.8, 200, "index flex 1"),
    7: (0.8, 200, "index flex 2"), 8: (0.8, 200, "index flex 3"),
    9: (1.2, 250, "middle spread"), 10: (0.8, 200, "middle flex 1"),
    11: (0.8, 200, "middle flex 2"), 12: (0.8, 200, "middle flex 3"),
    13: (1.2, 250, "ring spread"), 14: (0.8, 200, "ring flex 1"),
    15: (0.8, 200, "ring flex 2"), 16: (0.8, 200, "ring flex 3"),
    17: (1.5, 320, "pinky spread"), 18: (1.1, 260, "pinky flex 1"),
    19: (0.8, 200, "pinky flex 2"), 20: (0.8, 200, "pinky flex 3"),
}

# Motion parameters - Enhanced for Motor2
FLEX_DEG_DEFAULT = 90.0
FLEX_DEG_THUMB_CMC = 85.0  # Motor 2: CMC flexion range
FLEX_DEG_THUMB_MCP = 100.0  # Motor 3: MCP flexion range
FLEX_DEG_THUMB_IP = 90.0  # Motor 4: IP flexion range

# Motor2 position-based control parameters
MOTOR2_DISTANCE_MIN = 0.08  # 완전히 접힌 상태 (정규화된 거리)
MOTOR2_DISTANCE_MAX = 0.28  # 완전히 펼쳐진 상태
MOTOR2_WEIGHT_DISTANCE = 0.6  # 거리 기반 가중치
MOTOR2_WEIGHT_ANGLE = 0.4  # 각도 기반 가중치

SPLAY_GAIN_DEFAULT = 1.0
SPLAY_LIMIT_DEFAULT_DEG = 25.0
SPLAY_GAIN_THUMB = 2.0
SPLAY_LIMIT_THUMB_DEG = 90.0
SMOOTH_ALPHA = 0.35

# Control limits
DEADBAND_0P1DEG = 8
MAX_DUTY_STEP = 40
TOTAL_DUTY_BUDGET = 1500
MAX_ACTIVE_JOINTS = 12
MIN_DUTY_TO_MOVE = 18
PROTECTED_JOINTS = {17, 18}

# Mappings
FINGER_ORDER = ["finger1", "finger2", "finger3", "finger4", "finger5"]
JOINT_MAP = {
    "finger1": [1, 2, 3, 4], "finger2": [5, 6, 7, 8], "finger3": [9, 10, 11, 12],
    "finger4": [13, 14, 15, 16], "finger5": [17, 18, 19, 20]
}
TARGET_SIGN = {i: 1 for i in range(1, 21)}
for m in [3, 4, 17, 18]:
    TARGET_SIGN[m] = -1
DUTY_SIGN = {i: 1 for i in range(1, 21)}

# Motor state
MOTOR_ENABLED = {m: True for m in range(1, 21)}


# -----------------------------
# Helper functions
# -----------------------------
def make_zero_duty():
    return {m: 0 for m in range(1, 21)}


def disabled_motor_text():
    disabled = [m for m in range(1, 21) if not MOTOR_ENABLED[m]]
    return ",".join(str(m) for m in disabled) if disabled else "None"


def toggle_motor_enable(motor_id, cur_pos, prev_target, prev_duty):
    if not (1 <= motor_id <= 20):
        return f"Invalid motor id: {motor_id}"
    MOTOR_ENABLED[motor_id] = not MOTOR_ENABLED[motor_id]
    hold = int(cur_pos.get(motor_id, prev_target.get(motor_id, 0)))
    prev_target[motor_id] = hold
    prev_duty[motor_id] = 0
    status = "OFF" if not MOTOR_ENABLED[motor_id] else "ON"
    role = MOTOR_CONFIG[motor_id][2]
    return f"Motor {motor_id:02d} ({role}) -> {status}"


def toggle_all_motors(cur_pos, prev_target, prev_duty):
    any_enabled = any(MOTOR_ENABLED[m] for m in range(1, 21))
    new_state = not any_enabled
    for m in range(1, 21):
        MOTOR_ENABLED[m] = new_state
        if not new_state:
            hold = int(cur_pos.get(m, 0))
            prev_target[m] = hold
            prev_duty[m] = 0
    return f"All motors -> {'ON' if new_state else 'OFF'}"


def toggle_finger_motors(finger_num, cur_pos, prev_target, prev_duty):
    if not (1 <= finger_num <= 5):
        return f"Invalid finger number: {finger_num}"
    finger_key = f"finger{finger_num}"
    motor_ids = JOINT_MAP[finger_key]
    finger_names = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Ring", 5: "Pinky"}
    all_enabled = all(MOTOR_ENABLED[m] for m in motor_ids)
    new_state = not all_enabled
    for m in motor_ids:
        MOTOR_ENABLED[m] = new_state
        if not new_state:
            hold = int(cur_pos.get(m, 0))
            prev_target[m] = hold
            prev_duty[m] = 0
    finger_name = finger_names[finger_num]
    motor_list = ",".join(str(m) for m in motor_ids)
    return f"{finger_name} (M{motor_list}) -> {'ON' if new_state else 'OFF'}"


def enforce_motor_enable_mask(cur_pos, **kwargs):
    for m in range(1, 21):
        if MOTOR_ENABLED[m]:
            continue
        hold = int(cur_pos.get(m, 0))
        for key, data_dict in kwargs.items():
            if data_dict is not None:
                if key in ['desired', 'target']:
                    data_dict[m] = hold
                else:
                    data_dict[m] = 0


def clamp_target_0p1deg(motor_id, target_0p1deg):
    limits = {
        1: (-150, 290), 2: (-850, 900), 3: (-1500, 290), 4: (-900, 900),
        5: (-200, 310), 6: (0, 1150), 17: (-300, 0), 18: (-900, 150)
    }
    if motor_id in limits:
        lo, hi = limits[motor_id]
        return int(np.clip(int(target_0p1deg), lo, hi))
    return int(np.clip(int(target_0p1deg), -900, 1150))


def rate_limit_target(motor_id, desired, prev):
    max_speed = 80.0 if motor_id in [1, 5, 9, 13, 17] else 100.0
    if motor_id == 2:
        max_speed = 65.0  # CMC Flexion speed
    elif motor_id == 3:
        max_speed = 70.0
    elif motor_id == 4:
        max_speed = 90.0
    max_delta = max(1, int(max_speed * 10.0 * DT))
    delta = int(desired) - int(prev)
    return int(prev) + np.clip(delta, -max_delta, max_delta)


def clamp_step_to_current(motor_id, desired, current):
    max_step = 100 if motor_id in [1, 5, 9, 13, 17] else 250
    if motor_id == 2:
        max_step = 130  # CMC Flexion step
    elif motor_id == 3:
        max_step = 150
    elif motor_id == 4:
        max_step = 200
    return int(np.clip(int(desired), int(current) - max_step, int(current) + max_step))


def slew_limit_duty(motor_id, new_duty, prev_duty):
    prev = int(prev_duty.get(motor_id, 0))
    nd = int(new_duty)
    limited = int(np.clip(nd, prev - MAX_DUTY_STEP, prev + MAX_DUTY_STEP))
    prev_duty[motor_id] = limited
    return limited


def apply_global_limits(raw_duty_dict):
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
        duty = {m: int(v * scale) for m, v in duty.items()}
    return duty


def set_camera_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return actual_width, actual_height


# -----------------------------
# Enhanced thumb position calculation
# -----------------------------
def compute_thumb_cmc_position(lms_np):
    """
    Motor2 전용 하이브리드 포지션 제어:
    3D 거리 기반 + 관절 각도 기반을 결합
    """
    # 거리 기반 계산
    thumb_tip = lms_np[4]
    wrist = lms_np[0]
    index_mcp = lms_np[5]

    thumb_distance = np.linalg.norm(thumb_tip - wrist)
    hand_size = np.linalg.norm(index_mcp - wrist)
    if hand_size < 1e-6:
        hand_size = 1.0

    normalized_distance = thumb_distance / hand_size
    normalized_distance = np.clip(normalized_distance, MOTOR2_DISTANCE_MIN, MOTOR2_DISTANCE_MAX)
    distance_ratio = (normalized_distance - MOTOR2_DISTANCE_MIN) / (MOTOR2_DISTANCE_MAX - MOTOR2_DISTANCE_MIN)
    distance_ratio = float(np.clip(distance_ratio, 0.0, 1.0))

    # 각도 기반 계산 (CMC 관절)
    p0, p1, p2 = lms_np[0], lms_np[1], lms_np[2]
    ang_cmc = _angle_deg(p0 - p1, p2 - p1)
    angle_ratio = _curl_from_joint_angle(ang_cmc, open_deg=155.0, closed_deg=95.0)

    # 하이브리드 결합
    hybrid_ratio = (MOTOR2_WEIGHT_DISTANCE * distance_ratio +
                    MOTOR2_WEIGHT_ANGLE * angle_ratio)

    return float(np.clip(hybrid_ratio, 0.0, 1.0))


# -----------------------------
# Network client (동일)
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
            val = struct.unpack(">h", payload[i + 1:i + 3])[0]
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
# Hand tracking - Enhanced
# -----------------------------
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


def _finger_curl(lms, finger_idx):
    finger_landmarks = {
        "finger1": [1, 2, 3, 4], "finger2": [5, 6, 7, 8], "finger3": [9, 10, 11, 12],
        "finger4": [13, 14, 15, 16], "finger5": [17, 18, 19, 20]
    }
    idx = finger_landmarks[finger_idx]
    pts = [lms[i] for i in idx]
    if finger_idx == "finger1":
        return 0.0  # 엄지는 별도 처리
    mcp, pip, dip, tip = pts
    ang_pip = _angle_deg(mcp - pip, dip - pip)
    ang_dip = _angle_deg(pip - dip, tip - dip)
    avg = 0.6 * ang_pip + 0.4 * ang_dip
    return _curl_from_joint_angle(avg)


def _thumb_joint_curls(lms):
    """
    엄지의 3개 관절 각각 계산:
    Motor2: CMC (하이브리드 방식으로 별도 처리)
    Motor3: MCP
    Motor4: IP
    """
    p1, p2, p3, p4 = lms[1], lms[2], lms[3], lms[4]

    # MCP Flexion (Motor 3)
    ang_mcp = _angle_deg(p1 - p2, p3 - p2)
    curl_mcp = _curl_from_joint_angle(ang_mcp)

    # IP Flexion (Motor 4)
    ang_ip = _angle_deg(p2 - p3, p4 - p3)
    curl_ip = _curl_from_joint_angle(ang_ip)

    return curl_mcp, curl_ip


def _draw_landmarks(frame, lms_xy):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
    ]
    for a, b in connections:
        cv2.line(frame, lms_xy[a], lms_xy[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(lms_xy):
        color = (255, 0, 0) if i == 0 else ((0, 255, 255) if i in [4, 8, 12, 16, 20] else (0, 0, 255))
        radius = 7 if i == 0 else (6 if i in [4, 8, 12, 16, 20] else 4)
        cv2.circle(frame, (x, y), radius, color, -1)


def compute_splay_deg(lms_np):
    def finger_dir_2d(mcp_idx, tip_idx):
        v = lms_np[tip_idx] - lms_np[mcp_idx]
        return np.array([v[0], v[1]], dtype=np.float32)

    def signed_angle_2d(a, b):
        def unit2(v):
            n = float(np.linalg.norm(v))
            return v / (n + 1e-9)

        a = unit2(a)
        b = unit2(b)
        return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))

    dirs = {
        "finger2": finger_dir_2d(5, 8), "finger3": finger_dir_2d(9, 12),
        "finger4": finger_dir_2d(13, 16), "finger5": finger_dir_2d(17, 20),
        "finger1": finger_dir_2d(2, 4)
    }
    base = dirs["finger3"]
    return {
        "finger3": 0.0,
        "finger2": signed_angle_2d(base, dirs["finger2"]),
        "finger4": signed_angle_2d(base, dirs["finger4"]),
        "finger5": signed_angle_2d(base, dirs["finger5"]),
        "finger1": signed_angle_2d(base, dirs["finger1"]),
    }


class HandTrackerTasks:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("[OK] Hand tracker ready")

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_img)
        if not res.hand_landmarks:
            return frame_bgr, None, None, None, None, None

        hand_idx = None
        if hasattr(res, "handedness") and res.handedness:
            for idx, handed_list in enumerate(res.handedness):
                if handed_list and len(handed_list) > 0:
                    if handed_list[0].category_name == "Right":
                        hand_idx = idx
                        break

        if hand_idx is None:
            return frame_bgr, None, None, None, None, None

        lms = res.hand_landmarks[hand_idx]
        lms_np = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        lms_xy = [(int(p.x * w), int(p.y * h)) for p in lms]
        _draw_landmarks(frame_bgr, lms_xy)

        curls = {f: _finger_curl(lms_np, f) for f in FINGER_ORDER}
        splay = compute_splay_deg(lms_np)

        # Enhanced thumb processing
        thumb_cmc_position = compute_thumb_cmc_position(lms_np)  # Motor2용
        thumb_mcp_curl, thumb_ip_curl = _thumb_joint_curls(lms_np)  # Motor3,4용

        return frame_bgr, curls, splay, (thumb_cmc_position, thumb_mcp_curl, thumb_ip_curl), lms_xy, thumb_cmc_position


# -----------------------------
# HUD functions (동일)
# -----------------------------
def cv_safe_text(text):
    if text is None: return ""
    replacements = {"°": "deg", "→": "->", "←": "<-", "✅": "[OK]", "⚠️": "[WARN]"}
    for k, v in replacements.items(): text = str(text).replace(k, v)
    return text.encode("ascii", "replace").decode("ascii")


def put_text_outline(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
    text = cv_safe_text(text)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_alpha_box(img, x1, y1, x2, y2, color=(40, 40, 40), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_runtime_hud(frame, status_text, lms_xy, cur_pos, target_dict, duty_dict,
                     emergency_stop, home_zero, motor_input_buffer, flash_text, resolution_text,
                     motor2_debug=None):
    status_color = (0, 0, 255) if emergency_stop else ((0, 165, 255) if home_zero else (0, 255, 0))
    if "error" in status_text.lower() or "show your hand" in status_text.lower():
        status_color = (0, 0, 255)
    put_text_outline(frame, status_text, (10, 30), scale=0.65, color=status_color, thickness=2)

    if lms_xy and cur_pos:
        offsets = [(-52, -10), (-52, 12), (8, -10), (8, 12)]
        for motor_id in range(1, min(21, len(lms_xy))):
            if motor_id in cur_pos:
                x, y = lms_xy[motor_id]
                dx, dy = offsets[(motor_id - 1) % 4]
                cur_deg = cur_pos[motor_id] / 10.0
                color = (0, 230, 255) if MOTOR_ENABLED[motor_id] else (120, 120, 120)
                if target_dict and motor_id in target_dict:
                    tgt_deg = target_dict[motor_id] / 10.0
                    label = f"M{motor_id:02d} C:{cur_deg:+.1f} T:{tgt_deg:+.1f}"
                else:
                    label = f"M{motor_id:02d} C:{cur_deg:+.1f}"
                put_text_outline(frame, label, (x + dx, y + dy), scale=0.36, color=color, thickness=1)

    h, w = frame.shape[:2]
    x1, y1 = w - 320, 10
    x2, y2 = w - 10, 270
    draw_alpha_box(frame, x1, y1, x2, y2, color=(35, 35, 35), alpha=0.62)
    put_text_outline(frame, "Current Duty", (x1 + 10, y1 + 22), scale=0.58, color=(255, 255, 255), thickness=1)

    total_abs = sum(abs(int(v)) for v in duty_dict.values())
    active = sum(1 for v in duty_dict.values() if int(v) != 0)
    put_text_outline(frame, f"sum|d|={total_abs} active={active}", (x1 + 10, y1 + 44), scale=0.45,
                     color=(180, 255, 180), thickness=1)

    for idx, motor_id in enumerate(range(1, 21)):
        col = 0 if idx < 10 else 1
        row = idx if idx < 10 else idx - 10
        x = x1 + 10 + col * 155
        y = y1 + 70 + row * 17
        v = int(duty_dict.get(motor_id, 0))
        label = f"M{motor_id:02d}: {v:+4d}"
        if not MOTOR_ENABLED[motor_id]: label += " OFF"
        color = ((0, 220, 255) if v > 0 else (255, 180, 0)) if MOTOR_ENABLED[motor_id] and v != 0 else (120, 120, 120)
        put_text_outline(frame, label, (x, y), scale=0.42, color=color, thickness=1)

    x1, y2 = 10, h - 10
    x2, y1 = 620, h - 245
    draw_alpha_box(frame, x1, y1, x2, y2, color=(35, 35, 35), alpha=0.58)
    put_text_outline(frame, "Keys", (x1 + 10, y1 + 22), scale=0.58, color=(255, 255, 255), thickness=1)

    lines = [
        "[Q] Quit",
        f"[X] EMO   : {'ON' if emergency_stop else 'OFF'}",
        f"[R] HOME0 : {'ON' if home_zero else 'OFF'}",
        "[T] Toggle ALL motors ON/OFF",
        "[+/-] Change resolution",
        "[0-9] + [Enter] Toggle single motor",
        "[1-5] + [F] Toggle finger group",
        f"Input : {motor_input_buffer if motor_input_buffer else '-'}",
        f"Disabled : {disabled_motor_text()}",
        f"Resolution : {resolution_text}",
    ]

    # Motor2 디버그 정보 추가
    if motor2_debug is not None:
        lines.append(f"M2 pos : {motor2_debug:.3f}")

    y = y1 + 45
    for line in lines:
        put_text_outline(frame, line, (x1 + 10, y), scale=0.42, color=(220, 220, 220), thickness=1)
        y += 18

    if flash_text:
        put_text_outline(frame, flash_text, (x1 + 10, y + 10), scale=0.48, color=(0, 255, 255), thickness=1)


def to_duty(err_0p1deg, motor_id):
    if abs(err_0p1deg) < DEADBAND_0P1DEG: return 0
    kp, lim, _ = MOTOR_CONFIG[motor_id]
    d = int(kp * err_0p1deg)
    d = int(np.clip(d, -lim, lim))
    return DUTY_SIGN[motor_id] * d


def curl_to_flex_deg(curl_now, flex_deg):
    return float(np.clip(curl_now, 0.0, 1.0)) * flex_deg


# -----------------------------
# Main function
# -----------------------------
def main():
    print("===================================================")
    print("DG-5F-M DevMode Teleop - Enhanced Motor2")
    print("keys: q quit | x EMO | r HOME0 | t toggle ALL")
    print("      +/- resolution | num + Enter: motor toggle")
    print("      num(1-5) + F: finger toggle")
    print("===================================================")

    gr = DG5FDevClient(GRIPPER_IP, GRIPPER_PORT, timeout=0.5)
    gr.connect()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        gr.close()
        raise RuntimeError("Camera open failed")

    resolution_index = DEFAULT_RESOLUTION_INDEX
    current_width, current_height = set_camera_resolution(
        cap, RESOLUTION_PRESETS[resolution_index][0], RESOLUTION_PRESETS[resolution_index][1]
    )
    print(f"[OK] Initial resolution: {current_width}x{current_height}")

    tracker = HandTrackerTasks()

    smooth_curl = {f: 0.0 for f in FINGER_ORDER}
    smooth_splay = {f: 0.0 for f in FINGER_ORDER}
    smooth_thumb_cmc = 0.0  # Motor2용
    smooth_thumb_mcp = 0.0  # Motor3용
    smooth_thumb_ip = 0.0  # Motor4용

    emergency_stop = False
    home_zero = False
    prev_target = {m: 0 for m in range(1, 21)}
    prev_target_valid = False
    prev_duty = {m: 0 for m in range(1, 21)}
    last_sent_duty = {m: 0 for m in range(1, 21)}
    last_target = {m: 0 for m in range(1, 21)}
    motor_input_buffer = ""
    flash_text = ""
    flash_t_end = 0.0

    def reset_targets_to_current(cur_pos):
        nonlocal prev_target, prev_target_valid, last_target
        for m in range(1, 21):
            v = int(cur_pos.get(m, 0))
            prev_target[m] = v
            last_target[m] = v
        prev_target_valid = True

    def reset_duty_state():
        for m in range(1, 21): prev_duty[m] = 0

    def change_resolution(direction):
        nonlocal resolution_index, current_width, current_height, flash_text, flash_t_end, cap, prev_target_valid
        new_index = resolution_index + direction
        if new_index < 0 or new_index >= len(RESOLUTION_PRESETS):
            flash_text = "Resolution limit reached"
            flash_t_end = time.time() + 1.5
            return
        resolution_index = new_index
        target_width, target_height = RESOLUTION_PRESETS[resolution_index]
        cap.release()
        time.sleep(0.1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash_text = "Camera reopen failed"
            flash_t_end = time.time() + 2.0
            return
        current_width, current_height = set_camera_resolution(cap, target_width, target_height)
        for _ in range(5): cap.grab()
        prev_target_valid = False
        flash_text = f"Resolution: {current_width}x{current_height}"
        flash_t_end = time.time() + 2.0
        print(f"[OK] Resolution changed to {current_width}x{current_height}")

    last = time.time()

    try:
        while True:
            now = time.time()
            if now - last < DT: time.sleep(max(0.0, DT - (now - last)))
            last = time.time()

            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # Enhanced tracking with Motor2 debug info
            frame, curls, splay, thumb_tuple, lms_xy, motor2_debug = tracker.process(frame)
            key = cv2.waitKey(1) & 0xFF

            if time.time() > flash_t_end: flash_text = ""
            if key == ord("q"): break

            # Resolution control
            if key == ord("+") or key == ord("="):
                change_resolution(1)
            elif key == ord("-") or key == ord("_"):
                change_resolution(-1)

            # Motor ID Input
            if ord('0') <= key <= ord('9'):
                if len(motor_input_buffer) < 2: motor_input_buffer += chr(key)
            if key in (8, 127): motor_input_buffer = motor_input_buffer[:-1]

            # Communication
            try:
                cur = gr.get_positions()
            except Exception as e:
                try:
                    gr.set_duty(make_zero_duty())
                except:
                    pass
                reset_duty_state()
                last_sent_duty = make_zero_duty()
                draw_runtime_hud(frame, f"Comm error: {e}", lms_xy, None, None, last_sent_duty,
                                 emergency_stop, home_zero, motor_input_buffer, flash_text,
                                 f"{current_width}x{current_height}", motor2_debug)
                cv2.imshow("DG-5F Dev Teleop", frame)
                prev_target_valid = False
                continue

            # Toggle Single Motor (Enter)
            if key in (10, 13):
                if motor_input_buffer:
                    motor_id = int(motor_input_buffer)
                    msg = toggle_motor_enable(motor_id, cur, prev_target, prev_duty)
                    flash_text = msg
                    flash_t_end = time.time() + 1.5
                    motor_input_buffer = ""

            # Toggle All Motors (T)
            if key == ord("t"):
                flash_text = toggle_all_motors(cur, prev_target, prev_duty)
                flash_t_end = time.time() + 2.0
                if "OFF" in flash_text:
                    gr.set_duty(make_zero_duty())
                    reset_duty_state()
                    last_sent_duty = make_zero_duty()

            # Toggle Finger Group (F)
            if key == ord("f") or key == ord("F"):
                if motor_input_buffer:
                    try:
                        finger_num = int(motor_input_buffer)
                        msg = toggle_finger_motors(finger_num, cur, prev_target, prev_duty)
                        flash_text = msg
                        flash_t_end = time.time() + 2.0
                    except:
                        flash_text = "Invalid finger number"
                        flash_t_end = time.time() + 1.5
                    motor_input_buffer = ""

            # Emergency/Home controls
            if key == ord("x"):
                emergency_stop = not emergency_stop
                if emergency_stop: home_zero = False
            if key == ord("r"):
                home_zero = not home_zero
                if home_zero: emergency_stop = False

            if not prev_target_valid:
                reset_targets_to_current(cur)
                reset_duty_state()

            resolution_text = f"{current_width}x{current_height}"

            # Emergency Stop
            if emergency_stop:
                gr.set_duty(make_zero_duty())
                reset_targets_to_current(cur)
                reset_duty_state()
                last_sent_duty = make_zero_duty()
                draw_runtime_hud(frame, "EMERGENCY STOP", lms_xy, cur, last_target, last_sent_duty,
                                 emergency_stop, home_zero, motor_input_buffer, flash_text,
                                 resolution_text, motor2_debug)
                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # Home Zero
            if home_zero:
                desired = {m: clamp_target_0p1deg(m, 0) for m in range(1, 21)}
                for m in range(1, 21): desired[m] = clamp_step_to_current(m, desired[m], cur.get(m, 0))
                enforce_motor_enable_mask(cur, desired=desired, prev_target=prev_target, prev_duty=prev_duty)
                target = {}
                for m in range(1, 21):
                    target[m] = rate_limit_target(m, desired[m], prev_target[m])
                    prev_target[m] = target[m]
                last_target = dict(target)
                enforce_motor_enable_mask(cur, target=target, prev_target=prev_target, prev_duty=prev_duty)
                raw = {m: to_duty(target[m] - cur.get(m, 0), m) for m in range(1, 21)}
                enforce_motor_enable_mask(cur, raw=raw)
                raw = apply_global_limits(raw)
                enforce_motor_enable_mask(cur, raw=raw)
                duty = {m: slew_limit_duty(m, raw[m], prev_duty) for m in range(1, 21)}
                enforce_motor_enable_mask(cur, duty=duty, prev_duty=prev_duty)
                gr.set_duty(duty)
                last_sent_duty = dict(duty)
                draw_runtime_hud(frame, "HOME0 (returning to zero)", lms_xy, cur, last_target, last_sent_duty,
                                 emergency_stop, home_zero, motor_input_buffer, flash_text,
                                 resolution_text, motor2_debug)
                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # No hand detected
            if curls is None or splay is None or thumb_tuple is None:
                gr.set_duty(make_zero_duty())
                reset_targets_to_current(cur)
                reset_duty_state()
                last_sent_duty = make_zero_duty()
                draw_runtime_hud(frame, "Show your hand", lms_xy, cur, last_target, last_sent_duty,
                                 emergency_stop, home_zero, motor_input_buffer, flash_text,
                                 resolution_text, motor2_debug)
                cv2.imshow("DG-5F Dev Teleop", frame)
                continue

            # Hand tracking and control
            # Smoothing
            for f in FINGER_ORDER:
                smooth_curl[f] = (1.0 - SMOOTH_ALPHA) * smooth_curl[f] + SMOOTH_ALPHA * curls[f]
                curls[f] = smooth_curl[f]
                smooth_splay[f] = (1.0 - SMOOTH_ALPHA) * smooth_splay[f] + SMOOTH_ALPHA * splay[f]
                splay[f] = smooth_splay[f]

            # Enhanced thumb smoothing
            t_cmc_pos, t_mcp, t_ip = thumb_tuple
            smooth_thumb_cmc = (1.0 - SMOOTH_ALPHA) * smooth_thumb_cmc + SMOOTH_ALPHA * t_cmc_pos
            smooth_thumb_mcp = (1.0 - SMOOTH_ALPHA) * smooth_thumb_mcp + SMOOTH_ALPHA * t_mcp
            smooth_thumb_ip = (1.0 - SMOOTH_ALPHA) * smooth_thumb_ip + SMOOTH_ALPHA * t_ip
            t_cmc_pos, t_mcp, t_ip = smooth_thumb_cmc, smooth_thumb_mcp, smooth_thumb_ip

            # Target calculation
            desired = {m: prev_target[m] for m in range(1, 21)}
            for f in FINGER_ORDER:
                j0, j1, j2, j3 = JOINT_MAP[f]
                if f == "finger1":
                    # Motor 1: Opposition/Spread
                    spread_deg = float(
                        np.clip(SPLAY_GAIN_THUMB * splay[f], -SPLAY_LIMIT_THUMB_DEG, SPLAY_LIMIT_THUMB_DEG))

                    # Motor 2: CMC Flexion (Enhanced position-based)
                    cmc_deg = curl_to_flex_deg(t_cmc_pos, FLEX_DEG_THUMB_CMC)

                    # Motor 3: MCP Flexion
                    mcp_deg = curl_to_flex_deg(t_mcp, FLEX_DEG_THUMB_MCP)

                    # Motor 4: IP Flexion
                    ip_deg = curl_to_flex_deg(t_ip, FLEX_DEG_THUMB_IP)

                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * int(spread_deg * 10))
                    desired[j1] = clamp_target_0p1deg(j1, TARGET_SIGN[j1] * int(cmc_deg * 10))
                    desired[j2] = clamp_target_0p1deg(j2, TARGET_SIGN[j2] * int(mcp_deg * 10))
                    desired[j3] = clamp_target_0p1deg(j3, TARGET_SIGN[j3] * int(ip_deg * 10))
                else:
                    # Other fingers (unchanged)
                    spread_deg = float(
                        np.clip(SPLAY_GAIN_DEFAULT * splay[f], -SPLAY_LIMIT_DEFAULT_DEG, SPLAY_LIMIT_DEFAULT_DEG))
                    flex_deg = curl_to_flex_deg(curls[f], FLEX_DEG_DEFAULT)
                    desired[j0] = clamp_target_0p1deg(j0, TARGET_SIGN[j0] * int(spread_deg * 10))
                    for jx in [j1, j2, j3]:
                        desired[jx] = clamp_target_0p1deg(jx, TARGET_SIGN[jx] * int(flex_deg * 10))

            # Control pipeline
            for m in range(1, 21): desired[m] = clamp_step_to_current(m, desired[m], cur.get(m, 0))
            enforce_motor_enable_mask(cur, desired=desired, prev_target=prev_target, prev_duty=prev_duty)

            target = {}
            for m in range(1, 21):
                target[m] = rate_limit_target(m, desired[m], prev_target[m])
                prev_target[m] = target[m]
            last_target = dict(target)
            enforce_motor_enable_mask(cur, target=target, prev_target=prev_target, prev_duty=prev_duty)

            raw = {m: to_duty(target[m] - cur.get(m, 0), m) for m in range(1, 21)}
            enforce_motor_enable_mask(cur, raw=raw)
            raw = apply_global_limits(raw)
            enforce_motor_enable_mask(cur, raw=raw)

            duty = {m: slew_limit_duty(m, raw.get(m, 0), prev_duty) for m in range(1, 21)}
            enforce_motor_enable_mask(cur, duty=duty, prev_duty=prev_duty)

            # Send commands
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

            status = f"TELEOP Enhanced M2 | Budget={TOTAL_DUTY_BUDGET} TopK={MAX_ACTIVE_JOINTS}"
            draw_runtime_hud(frame, status, lms_xy, cur, last_target, last_sent_duty,
                             emergency_stop, home_zero, motor_input_buffer, flash_text,
                             resolution_text, motor2_debug)
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
