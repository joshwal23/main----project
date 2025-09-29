from ultralytics import YOLO
import numpy as np
import cv2
import math
import random

model = YOLO("yolov8n-pose.pt")

def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle
    except:
        return 180

def get_body_parts(kpts):
    return {
        'L_Shoulder': kpts[5], 'R_Shoulder': kpts[6],
        'L_Elbow': kpts[7], 'R_Elbow': kpts[8],
        'L_Hip': kpts[11], 'R_Hip': kpts[12],
        'L_Knee': kpts[13], 'L_Ankle': kpts[15],
    }

STATE = {
    'exercise': 'unknown',
    'in_progress': False,
    'min_angle': 180,
    'reps': 0,
    'good': 0,
    'bad': 0
}

GAME = {
    'level': 1,
    'xp': 0,
    'xp_needed': 100,
    'message': ""
}

def update_game():
    s, g = STATE, GAME
    g['xp'] = s['good'] * 10 - s['bad'] * 2
    if g['xp'] < 0: g['xp'] = 0

    while g['xp'] >= g['xp_needed']:
        g['xp'] -= g['xp_needed']
        g['level'] += 1
        g['xp_needed'] = int(g['xp_needed'] * 1.2)
        g['message'] = random.choice([
            "🔥 Level Up! Keep pushing!",
            "💪 You’re getting stronger!",
            "⚡ Power unlocked!",
            "🏆 Champion in the making!"
        ])

def classify(parts):
    try:
        squat = calculate_angle(parts['L_Hip'], parts['L_Knee'], parts['L_Ankle'])
        pushup = calculate_angle(parts['L_Shoulder'], parts['L_Elbow'], parts['L_Hip'])
        situp = calculate_angle(parts['L_Shoulder'], parts['L_Hip'], parts['L_Knee'])
    except:
        return "unknown"

    variation = {
        "squat": abs(180 - squat),
        "pushup": abs(180 - pushup),
        "situp": abs(180 - situp),
    }
    return max(variation, key=variation.get)

def analyze_squat(parts):
    s = STATE
    angle = calculate_angle(parts['L_Hip'], parts['L_Knee'], parts['L_Ankle'])
    if angle < 140 and not s['in_progress']:
        s['in_progress'] = True
        s['min_angle'] = angle
    if s['in_progress']:
        s['min_angle'] = min(s['min_angle'], angle)
    if angle > 160 and s['in_progress']:
        s['in_progress'] = False
        s['reps'] += 1
        if 40 <= s['min_angle'] <= 120: s['good'] += 1
        else: s['bad'] += 1
        s['min_angle'] = 180
        update_game()

def analyze_pushup(parts):
    s = STATE
    angle = calculate_angle(parts['L_Shoulder'], parts['L_Elbow'], parts['L_Hip'])
    if angle < 160 and not s['in_progress']:
        s['in_progress'] = True
        s['min_angle'] = angle
    if s['in_progress']:
        s['min_angle'] = min(s['min_angle'], angle)
    if angle > 170 and s['in_progress']:
        s['in_progress'] = False
        s['reps'] += 1
        if 90 <= s['min_angle'] <= 120: s['good'] += 1
        else: s['bad'] += 1
        s['min_angle'] = 180
        update_game()

def analyze_situp(parts):
    s = STATE
    angle = calculate_angle(parts['L_Shoulder'], parts['L_Hip'], parts['L_Knee'])
    if angle < 160 and not s['in_progress']:
        s['in_progress'] = True
        s['min_angle'] = angle
    if s['in_progress']:
        s['min_angle'] = min(s['min_angle'], angle)
    if angle > 170 and s['in_progress']:
        s['in_progress'] = False
        s['reps'] += 1
        if 90 <= s['min_angle'] <= 120: s['good'] += 1
        else: s['bad'] += 1
        s['min_angle'] = 180
        update_game()

def draw_bar(img, x, y, w, h, value, max_value):
    ratio = value / max_value
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), (0, 255, 0), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color, shadow = (255, 255, 255), (0, 0, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, verbose=False)
        exercise = "UNKNOWN"

        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy.cpu().numpy()) > 0:
                kpts = r.keypoints.xy.cpu().numpy()[0]
                parts = get_body_parts(kpts)

                ex = classify(parts)
                STATE['exercise'] = ex
                exercise = ex.upper()

                if ex == "squat": analyze_squat(parts)
                elif ex == "pushup": analyze_pushup(parts)
                elif ex == "situp": analyze_situp(parts)

                frame = r.plot(conf=False, labels=False, boxes=False)

        def draw_text(img, text, pos, color, shadow):
            cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font, 0.7, shadow, 4, cv2.LINE_AA)
            cv2.putText(img, text, pos, font, 0.7, color, 2, cv2.LINE_AA)

        y = 30
        draw_text(frame, f"Exercise: {exercise}", (10, y), text_color, shadow); y += 30
        s, g = STATE, GAME
        draw_text(frame, f"Reps: {s['reps']} | Good: {s['good']} | Bad: {s['bad']}", (10, y), text_color, shadow); y += 30
        draw_text(frame, f"Level: {g['level']}", (10, y), (0, 255, 0), shadow); y += 30
        draw_text(frame, f"XP: {g['xp']}/{g['xp_needed']}", (10, y), (0, 200, 255), shadow)
        draw_bar(frame, 10, y + 10, 200, 20, g['xp'], g['xp_needed'])

        if g['message']:
            draw_text(frame, g['message'], (10, y + 60), (0, 255, 255), shadow)

        cv2.imshow("Fitness Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    s, g = STATE, GAME
    print("\n====== FINAL REPORT ======")
    print(f"{s['exercise'].upper()} → Reps: {s['reps']} | Good: {s['good']} | Bad: {s['bad']}")
    print(f"LEVEL: {g['level']} | XP: {g['xp']}/{g['xp_needed']}")

if __name__ == "__main__":
    process_video("exercise_video.mp4")
