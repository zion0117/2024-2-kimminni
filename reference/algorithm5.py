import cv2
import mediapipe as mp
import json
import numpy as np

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 기준 JSON 데이터 불러오기
with open("C:/Users/darwi/capstone_mindain/landmark/landmark5.json", "r") as f:
    reference_landmarks = json.load(f)

# 기준 관절 좌표 데이터 처리
def get_reference_coordinates(reference_data):
    coordinates = {}
    for lm in reference_data[0]:  # 첫 번째 프레임의 관절 좌표 리스트 사용
        joint_id = lm["id"]
        coordinates[joint_id] = (lm["x"], lm["y"])
    return coordinates

reference_coordinates = get_reference_coordinates(reference_landmarks)

# 주요 관절명 리스트 (영어)
joint_names = {
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle"
}

# 주요 관절 인덱스 (운동에 필요한 관절점)
key_joints = list(joint_names.keys())

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 벡터 계산
    ba = a - b
    bc = c - b

    # 내적과 벡터 크기를 이용하여 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# 사용자 웹캠에서 관절 좌표 추출 및 비교
cap = cv2.VideoCapture(0)  # 웹캠 열기

if not cap.isOpened():
    print("Error: 웹캠을 열 수 없습니다.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # 사용자 관절 좌표 추출
        user_landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]

        # 피드백 메시지 초기화
        feedback_messages = []

        # 관절 각도 비교
        # 어깨-팔꿈치-손목 각도 비교
        left_elbow_angle_user = calculate_angle(user_landmarks[11], user_landmarks[13], user_landmarks[15])
        right_elbow_angle_user = calculate_angle(user_landmarks[12], user_landmarks[14], user_landmarks[16])
        
        # 엉덩이-무릎-발목 각도 비교
        left_knee_angle_user = calculate_angle(user_landmarks[23], user_landmarks[25], user_landmarks[27])
        right_knee_angle_user = calculate_angle(user_landmarks[24], user_landmarks[26], user_landmarks[28])

        # 기준 각도 설정
        left_elbow_angle_ref = calculate_angle(reference_coordinates[11], reference_coordinates[13], reference_coordinates[15])
        right_elbow_angle_ref = calculate_angle(reference_coordinates[12], reference_coordinates[14], reference_coordinates[16])
        left_knee_angle_ref = calculate_angle(reference_coordinates[23], reference_coordinates[25], reference_coordinates[27])
        right_knee_angle_ref = calculate_angle(reference_coordinates[24], reference_coordinates[26], reference_coordinates[28])

        # 각도 비교 (허용 오차 범위: 10도)
        if abs(left_elbow_angle_user - left_elbow_angle_ref) > 10:
            feedback_messages.append("Left Elbow")
        if abs(right_elbow_angle_user - right_elbow_angle_ref) > 10:
            feedback_messages.append("Right Elbow")
        if abs(left_knee_angle_user - left_knee_angle_ref) > 10:
            feedback_messages.append("Left Knee")
        if abs(right_knee_angle_user - right_knee_angle_ref) > 10:
            feedback_messages.append("Right Knee")

        # 피드백 메시지 생성
        feedback_text = ", ".join(feedback_messages) if feedback_messages else "All poses are correct."

        # 웹캠 화면에 피드백 메시지 표시 (텍스트 크기 축소)
        cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 주요 관절점 및 선 연결 표시
        for joint_id in key_joints:
            x, y = int(user_landmarks[joint_id][0]), int(user_landmarks[joint_id][1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 초록색 점 표시

        # 주요 관절 연결 (운동에 필요한 선 + 몸통 연결 추가)
        connections = [
            (11, 13), (13, 15),  # 왼쪽 어깨-팔꿈치-손목
            (12, 14), (14, 16),  # 오른쪽 어깨-팔꿈치-손목
            (23, 25), (25, 27),  # 왼쪽 엉덩이-무릎-발목
            (24, 26), (26, 28),  # 오른쪽 엉덩이-무릎-발목
            (11, 23), (12, 24),  # 어깨-엉덩이 (몸통)
            (11, 12), (23, 24)   # 왼쪽-오른쪽 어깨, 왼쪽-오른쪽 엉덩이 (몸통 연결)
        ]

        for start, end in connections:
            start_x, start_y = int(user_landmarks[start][0]), int(user_landmarks[start][1])
            end_x, end_y = int(user_landmarks[end][0]), int(user_landmarks[end][1])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # 파란색 선 표시

        # 웹캠 영상 표시
        cv2.imshow("User Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
