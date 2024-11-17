import cv2
import mediapipe as mp
import numpy as np

# MediaPipe �ʱ�ȭ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ���� ���ϰ� ��ķ �ʱ�ȭ
video_path = 'C:/Users/user/My project/Assets/video(sample)/video.mp4'
cap_video = cv2.VideoCapture(video_path)
cap_webcam = cv2.VideoCapture(0)  # ��ķ �ʱ�ȭ

frame_count = 0  # ������ ī��Ʈ
pause_video = False  # ���� �Ͻ����� ����

def calculate_similarity(landmarks_video, landmarks_webcam):
    if not landmarks_video or not landmarks_webcam:
        return 0  # ���帶ũ�� ������ ���絵 0

    # �� ���帶ũ ��ǥ �� �Ÿ� ���
    distances = [
        np.linalg.norm(
            np.array([landmark_video.x, landmark_video.y, landmark_video.z]) -
            np.array([landmark_webcam.x, landmark_webcam.y, landmark_webcam.z])
        )
        for landmark_video, landmark_webcam in zip(landmarks_video, landmarks_webcam)
    ]
    avg_distance = np.mean(distances)

    # ���絵 ������ ��ȯ (0~100, �Ÿ��� �������� ���� ����)
    max_distance = 2  # �ִ� �Ÿ�, ��Ȳ�� �°� ����
    score = max(0, 100 - (avg_distance / max_distance) * 100)
    return round(score, 2)

def provide_feedback(landmarks_video, landmarks_webcam):
    feedback_messages = []

    # ���÷� ����� ������ ���� �� �ǵ�� ����
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
            break  # ������ ������ ����

    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_webcam:
        break  # ��ķ �������� ������ ����

    frame_count += 1

    # ��ķ ������ �¿� ����
    frame_webcam = cv2.flip(frame_webcam, 1)

    # ��ķ ������ �ػ�
    webcam_height, webcam_width = frame_webcam.shape[:2]

    # ���� �������� ��ķ ������ ũ�⿡ �°� ���� �����ϸ� ����
    scale_ratio = min(webcam_width / frame_video.shape[1], webcam_height / frame_video.shape[0])
    new_width = int(frame_video.shape[1] * scale_ratio)
    new_height = int(frame_video.shape[0] * scale_ratio)
    resized_video_frame = cv2.resize(frame_video, (new_width, new_height))

    # ������ �׵θ� �߰�
    top_border = (webcam_height - new_height) // 2
    bottom_border = webcam_height - new_height - top_border
    left_border = (webcam_width - new_width) // 2
    right_border = webcam_width - new_width - left_border
    bordered_video_frame = cv2.copyMakeBorder(
        resized_video_frame, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # MediaPipe Pose ó��
    rgb_frame_video = cv2.cvtColor(bordered_video_frame, cv2.COLOR_BGR2RGB)
    rgb_frame_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)

    results_video = pose.process(rgb_frame_video)
    results_webcam = pose.process(rgb_frame_webcam)

    # ���絵 ��� �� �ǵ�� ����
    similarity_score = 0
    if results_video.pose_landmarks and results_webcam.pose_landmarks:
        similarity_score = calculate_similarity(results_video.pose_landmarks.landmark, results_webcam.pose_landmarks.landmark)

        # ���� �ؽ�Ʈ ǥ��
        score_text = f"Similarity Score: {similarity_score}"
        cv2.putText(frame_webcam, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 90�� �̸��� �� �ǵ�� �޽��� ���
        if similarity_score < 90:
            feedback_messages = provide_feedback(results_video.pose_landmarks.landmark, results_webcam.pose_landmarks.landmark)
            for idx, message in enumerate(feedback_messages):
                cv2.putText(frame_webcam, message, (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 80�� �̸��� �� ���� �Ͻ� ����
        pause_video = similarity_score < 80

        # ���� �����Ӱ� ��ķ �����ӿ� ���帶ũ �׸���
        mp_drawing.draw_landmarks(bordered_video_frame, results_video.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_webcam, results_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # �� �������� �������� ��ġ��
    combined_frame = cv2.hconcat([bordered_video_frame, frame_webcam])
    cv2.imshow("Pose Detection - Video and Webcam", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ������ ��ķ ĸó ����
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()
