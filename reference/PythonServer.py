from flask import Flask, request, jsonify

app = Flask(__name__)

# 신체 부위에 해당하는 운동 리스트
exercise_mapping = {
    "chest": ["Push-ups", "Bench Press", "Chest Fly"],
    "shoulders": ["Shoulder Press", "Lateral Raise", "Front Raise"],
    "abs": ["Crunches", "Plank", "Leg Raises"],
    "back": ["Pull-ups", "Deadlift", "Rows"],
    "arms": ["Bicep Curls", "Tricep Dips", "Hammer Curls"],
    "thighs": ["Squats", "Lunges", "Leg Press"],
    "ankles": ["Ankle Circles", "Toe Raises", "Calf Raises"],
    "waist": ["Side Crunches", "Russian Twists", "Oblique Leg Raises"],
    "calves": ["Calf Raises", "Jump Rope", "Box Jumps"]
}

@app.route('/get_exercises', methods=['POST'])
def get_exercises():
    data = request.json
    body_parts = data.get('body_parts', [])  # 리스트로 받아오기

    # 선택된 부위에 맞는 운동을 모두 합쳐서 반환
    all_exercises = []
    for part in body_parts:
        exercises = exercise_mapping.get(part, [])
        all_exercises.extend(exercises)
    
    # 중복 제거를 위해 set() 사용 (필요하면)
    all_exercises = list(set(all_exercises))
    
    return jsonify({"exercises": all_exercises})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
