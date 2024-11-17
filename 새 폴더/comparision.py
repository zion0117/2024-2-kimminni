import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 비디오 파일과 웹캠 초기화
video_path = 'C:/Users/user/My project/Assets/video(sample)/video.mp4'
cap_video = cv2.VideoCapture(video_path)
cap_webcam = cv2.VideoCapture(0)  # 웹캠 초기화

frame_count = 0  # 프레임 카운트
pause_video = False  # 비디오 일시정지 상태

def calculate_similarity(landmarks_video, landmarks_webcam):
    if not landmarks_video or not landmarks_webcam:
        return 0  # 랜드마크가 없으면 유사도 0

    # 각 랜드마크 좌표 간 거리 계산
    distances = [
        np.linalg.norm(
            np.array([landmark_video.x, landmark_video.y, landmark_video.z]) -
            np.array([landmark_webcam.x, landmark_webcam.y, landmark_webcam.z])
        )
        for landmark_video, landmark_webcam in zip(landmarks_video, landmarks_webcam)
    ]
    avg_distance = np.mean(distances)

    # 유사도 점수로 변환 (0~100, 거리가 작을수록 높은 점수)
    max_distance = 2  # 최대 거리, 상황에 맞게 조정
    score = max(0, 100 - (avg_distance / max_distance) * 100)
    return round(score, 2)

def provide_feedback(landmarks_video, landmarks_webcam):
    feedback_messages = []

    # 예시로 어깨와 엉덩이 높이 비교 피드백 제공
    shoulder_video = landmarks_video[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder_webcam = landmarks_webcam[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip_video = landmarks_video[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_webcam = landmarks_webcam[mp_pose.PoseLandmark.LEFT_HIP.value]

    if abs(shoulder_video.y - shoulder_webcam.y) > 0.1:
        feedback_messages.append("move your shoulder .")
    
    if abs(hip_video.y - hip_webcam.y) > 0.1:
        feedback_messages.append("move your hip")
    
    return feedback_messages

while cap_video.isOpened() and cap_webcam.isOpened():
    if not pause_video:
        ret_video, frame_video = cap_video.read()
        if not ret_video:
            break  # 비디오가 끝나면 종료

    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_webcam:
        break  # 웹캠 프레임이 없으면 종료

    frame_count += 1

    # 웹캠 프레임 좌우 반전
    frame_webcam = cv2.flip(frame_webcam, 1)

    # 웹캠 프레임 해상도
    webcam_height, webcam_width = frame_webcam.shape[:2]

    # 비디오 프레임을 웹캠 프레임 크기에 맞게 비율 유지하며 조정
    scale_ratio = min(webcam_width / frame_video.shape[1], webcam_height / frame_video.shape[0])
    new_width = int(frame_video.shape[1] * scale_ratio)
    new_height = int(frame_video.shape[0] * scale_ratio)
    resized_video_frame = cv2.resize(frame_video, (new_width, new_height))

    # 검은색 테두리 추가
    top_border = (webcam_height - new_height) // 2
    bottom_border = webcam_height - new_height - top_border
    left_border = (webcam_width - new_width) // 2
    right_border = webcam_width - new_width - left_border
    bordered_video_frame = cv2.copyMakeBorder(
        resized_video_frame, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # MediaPipe Pose 처리
    rgb_frame_video = cv2.cvtColor(bordered_video_frame, cv2.COLOR_BGR2RGB)
    rgb_frame_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)

    results_video = pose.process(rgb_frame_video)
    results_webcam = pose.process(rgb_frame_webcam)

    # 유사도 계산 및 피드백 제공
    similarity_score = 0
    if results_video.pose_landmarks and results_webcam.pose_landmarks:
        similarity_score = calculate_similarity(results_video.pose_landmarks.landmark, results_webcam.pose_landmarks.landmark)

        # 점수 텍스트 표시
        score_text = f"Similarity Score: {similarity_score}"
        cv2.putText(frame_webcam, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 90점 미만일 때 피드백 메시지 출력
        if similarity_score < 90:
            feedback_messages = provide_feedback(results_video.pose_landmarks.landmark, results_webcam.pose_landmarks.landmark)
            for idx, message in enumerate(feedback_messages):
                cv2.putText(frame_webcam, message, (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 80점 미만일 때 비디오 일시 정지
        pause_video = similarity_score < 80

        # 영상 프레임과 웹캠 프레임에 랜드마크 그리기
        mp_drawing.draw_landmarks(bordered_video_frame, results_video.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_webcam, results_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 두 프레임을 수평으로 합치기
    combined_frame = cv2.hconcat([bordered_video_frame, frame_webcam])
    cv2.imshow("Pose Detection - Video and Webcam", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오와 웹캠 캡처 종료
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()
