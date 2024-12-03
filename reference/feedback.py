import cv2
import mediapipe as mp
import numpy as np
import asyncio
import openai
from PIL import Image, ImageDraw, ImageFont
import time
import os

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 상태 변수 초기화
feedback_message = ""
last_feedback_time = 0
feedback_interval = 2  # 피드백 최소 간격(초)
last_state = None  # 이전 상태(작게, 적당히, 크게)
pending_feedback = None  # GPT 피드백 대기 상태

# 움직임 상태 변수
last_angles = None  # 이전 프레임의 각도들
no_movement_start_time = None  # 멈춤 시작 시간
movement_threshold = 5  # 움직임 감지 임계값
no_movement_threshold = 1  # 멈춘 상태를 판단하기 위한 시간(초)

# 어깨 회전 상태 기준
ROTATION_RANGES = {
    "작게": (0, 20),
    "적당히": (20, 40),
    "크게": (40, 90)
}

# 한글 텍스트 표시 함수
def put_korean_text(frame, text, position, font_path="malgun.ttf", font_size=20, color=(0, 255, 0)):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# 각도 계산 함수
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    radians = np.arctan2(c[1] - b[1], c[0] - b[1]) - np.arctan2(a[1] - b[1], a[0] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# 비동기 API 호출
async def generate_feedback_from_api(prompt):
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 어르신의 운동 자세를 교정해주는 가상캐릭터 '미니'야."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# 자세 분석 함수
def analyze_pose(landmarks):
    global feedback_message, last_feedback_time, last_state, pending_feedback, last_angles, no_movement_start_time

    # 양쪽 팔꿈치와 어깨의 각도 계산
    left_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    right_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )

    # 평균 회전 각도 계산
    avg_angle = (left_elbow_angle + right_elbow_angle) / 2

    # 움직임 감지
    if last_angles is not None:
        if abs(avg_angle - last_angles) < movement_threshold:
            if no_movement_start_time is None:
                no_movement_start_time = time.time()
            elif time.time() - no_movement_start_time > no_movement_threshold:
                feedback_message = ""
                return  # 멈춤 상태에서는 피드백 출력 안 함
        else:
            no_movement_start_time = None

    last_angles = avg_angle

    # 상태 감지
    for state, (min_angle, max_angle) in ROTATION_RANGES.items():
        if min_angle <= avg_angle <= max_angle:
            current_state = state
            break
    else:
        current_state = "잘못된 자세"

    # 상태 변화 시 피드백 제공
    if current_state != last_state:
        if current_state == "작게":
            feedback_message = "어깨를 더 크게 회전하세요."
        elif current_state == "적당히":
            feedback_message = "좋습니다! 부드럽게 움직이고 있어요."
        elif current_state == "크게":
            feedback_message = "어깨를 더 작게 회전하세요."
        elif current_state == "잘못된 자세":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            feedback_message = loop.run_until_complete(
                generate_feedback_from_api("사용자의 자세가 이상합니다. 교정 피드백을 간결하게 1줄로 제공해 주세요.")
            )
        last_state = current_state

# 실시간 자세 분석
cap = cv2.VideoCapture(0)
nth_frame = 30  # 30번째 프레임마다 자세 분석
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 화면 크기 조정
    frame = cv2.resize(frame, (1280, 720))

    frame_count += 1
    if frame_count % nth_frame != 0:
        frame = put_korean_text(frame, feedback_message, (50, 50), font_size=30, color=(0, 255, 0))
        cv2.imshow('Exercise Feedback', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # MediaPipe로 자세 추적
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        analyze_pose(result.pose_landmarks.landmark)

    frame = put_korean_text(frame, feedback_message, (50, 50), font_size=30, color=(0, 255, 0))
    cv2.imshow('Exercise Feedback', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
