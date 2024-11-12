import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
import time

def extract_pose_landmarks(results):
    """미디어 파이프 결과에서 포즈 랜드마크를 벡터로 추출합니다."""
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return None

def calculate_similarity(landmarks_video, landmarks_webcam):
    """두 랜드마크 벡터의 코사인 유사도를 100점 만점 기준으로 변환합니다."""
    if landmarks_video is None or landmarks_webcam is None:
        return 0.0  # 랜드마크가 없는 경우 유사도 0
    similarity = cosine_similarity([landmarks_video], [landmarks_webcam])[0][0]
    return max(0.0, min(100.0, similarity * 100))

def give_feedback(results_video, results_webcam):
    """피드백을 제공합니다."""
    feedback = ""

    if results_video.pose_landmarks and results_webcam.pose_landmarks:
        # 왼쪽, 오른쪽 어깨와 손목의 좌표 추출
        left_shoulder_video = results_video.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder_video = results_video.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist_video = results_video.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

        left_shoulder_webcam = results_webcam.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder_webcam = results_webcam.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist_webcam = results_webcam.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]

        # 피드백 조건 1: 어깨를 제대로 잡지 않았을 때
        # 어깨와 손목의 거리 계산
        def calculate_distance(point1, point2):
            return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

        shoulder_wrist_distance_video = calculate_distance(left_shoulder_video, left_wrist_video)
        shoulder_wrist_distance_webcam = calculate_distance(left_shoulder_webcam, left_wrist_webcam)

        if shoulder_wrist_distance_webcam > shoulder_wrist_distance_video * 1.5:
            feedback = "어깨를 제대로 잡으세요"

        # 피드백 조건 2: 팔을 돌리긴 하는데 회전이 너무 작은 경우
        wrist_movement_video = abs(left_wrist_video.x - left_shoulder_video.x)
        wrist_movement_webcam = abs(left_wrist_webcam.x - left_shoulder_webcam.x)

        if wrist_movement_webcam < wrist_movement_video * 0.5:
            feedback = "회전을 더 크게 해주세요"

    return feedback

def compare_video_and_webcam(video_path):
    # 미디어 파이프 설정
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # 동영상 파일 열기
    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    # 웹캠 열기
    cap_webcam = cv2.VideoCapture(0)
    if not cap_webcam.isOpened():
        print("웹캠을 열 수 없습니다.")
        cap_video.release()
        return

    last_feedback_time = time.time()  # 피드백 시간 추적

    while True:  # 사용자가 종료할 때까지 무한 반복
        if cap_video.get(cv2.CAP_PROP_POS_FRAMES) == cap_video.get(cv2.CAP_PROP_FRAME_COUNT):
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 동영상에서 프레임 읽기
        ret_video, frame_video = cap_video.read()
        ret_webcam, frame_webcam = cap_webcam.read()

        if not ret_video or not ret_webcam:
            break

        # 웹캠 프레임을 동영상 프레임 크기에 맞추기 위해 조정
        frame_webcam = cv2.resize(frame_webcam, (frame_video.shape[1], frame_video.shape[0]))

        # 미디어 파이프 동작 인식
        rgb_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
        rgb_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)

        results_video = pose.process(rgb_video)
        results_webcam = pose.process(rgb_webcam)

        # 랜드마크 벡터 추출
        landmarks_video = extract_pose_landmarks(results_video)
        landmarks_webcam = extract_pose_landmarks(results_webcam)

        # 유사도 계산 (100점 만점 기준)
        similarity = calculate_similarity(landmarks_video, landmarks_webcam)

        # 피드백을 2초마다 업데이트하기 위해 조건 추가
        current_time = time.time()
        feedback = ""
        if current_time - last_feedback_time >= 2:  # 2초마다 피드백 제공
            feedback = give_feedback(results_video, results_webcam)
            last_feedback_time = current_time  # 마지막 피드백 시간을 업데이트

        # 좌표 그리기
        if results_video.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame_video, results_video.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_webcam.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame_webcam, results_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 두 프레임을 좌우로 결합
        combined_frame = np.hstack((frame_video, frame_webcam))

        # 유사도와 피드백 표시
        cv2.putText(combined_frame, f"Similarity: {similarity:.2f}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if feedback:  # 피드백이 있을 경우 화면에 출력
            cv2.putText(combined_frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 화면에 표시
        cv2.imshow('Video vs Webcam', combined_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_video.release()
    cap_webcam.release()
    cv2.destroyAllWindows()

# 동영상 파일 경로 설정
video_path = r"C:/Users/user/Downloads/어깨잡고돌리기.mp4"
compare_video_and_webcam(video_path)
