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

# ... (기존 상수들과 함수들은 동일하게 유지) ...

# -----------------------------
# Resolution management
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
# HUD 함수 수정
# -----------------------------
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
# Main 함수 수정
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

    # ... (기존 변수 초기화 코드들) ...

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
            # ... (기존 루프 시작 코드) ...

            # 해상도 변경 키 처리
            if key == ord("+") or key == ord("="):
                change_resolution(1)
            elif key == ord("-") or key == ord("_"):
                change_resolution(-1)

            # ... (기존 키 처리 코드들) ...

            # 모든 draw_runtime_hud 호출에 resolution_text 추가
            resolution_text = f"{current_width}x{current_height}"
            
            # 예시: EMO 상태에서의 HUD 호출
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

            # ... (나머지 모든 draw_runtime_hud 호출에도 resolution_text 매개변수 추가) ...

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
