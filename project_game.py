from ultralytics import YOLO
import cv2
import numpy as np
import random
import math

model = YOLO("yolov8n-pose.pt")

def angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = abs(r * 180 / math.pi)
        ang = 360 - ang if ang > 180 else ang
        return min(max(ang, 45), 180)
    except:
        return 180

def parts(k):
    return {'LS': k[5], 'LE': k[7], 'LH': k[11], 'LK': k[13], 'LA': k[15]}

STATE = {
    "exercise": "none",
    "in_rep": False,
    "min_angle": 180,
    "reps": 0
}

PLAYER = {"x": 120, "y": 300, "vy": 0}
GROUND = 300
OBSTACLES = []
TEXTS = []

GAME = {"score": 0, "slow": 0, "speed": 6}

class Obstacle:
    def __init__(self, w):
        self.x = w + random.randint(0, 200)
        self.y = GROUND
        self.r = random.choice([20, 30])
        self.color = random.choice([(0,0,255), (0,255,255)])
    def move(self, s):
        self.x -= s
    def draw(self, f):
        cv2.circle(f, (int(self.x), self.y), self.r, self.color, -1)

def popup(t):
    TEXTS.append({"txt": t, "life": 30, "y": 200})

def update_game(ok):
    if ok:
        GAME["score"] += 10
        PLAYER["vy"] = -18
        popup("PERFECT!")
        if GAME["score"] % 40 == 0:
            GAME["slow"] = 20
    else:
        GAME["score"] = max(0, GAME["score"] - 2)
        popup("OK")
        OBSTACLES.append(Obstacle(640))

def analyze(a, down, up, good):
    s = STATE

    # START REP (very easy)
    if a < down and not s["in_rep"]:
        s["in_rep"] = True
        s["min_angle"] = a

    # TRACK LOWEST POINT
    if s["in_rep"]:
        s["min_angle"] = min(s["min_angle"], a)

    # END REP (soft exit)
    if a > up and s["in_rep"]:
        s["in_rep"] = False
        s["reps"] += 1
        update_game(good[0] <= s["min_angle"] <= good[1])
        s["min_angle"] = 180

print("\nSelect Input Source")
print("1 - Live Camera")
print("2 - Video File")
choice = input("Enter choice (1/2): ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    path = input("Enter full video file path: ").strip()
    cap = cv2.VideoCapture(path)
else:
    exit()

if not cap.isOpened():
    print("Error opening video source")
    exit()

cv2.namedWindow("Camera")
cv2.namedWindow("Game")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    game = np.zeros((480, 640, 3), dtype=np.uint8)
    game[:] = (30, 30, 30)

    results = model.predict(frame, verbose=False)

    for r in results:
        if r.keypoints is None:
            continue

        k = r.keypoints.xy.cpu().numpy()[0]
        p = parts(k)

        squat = angle(p['LH'], p['LK'], p['LA'])
        push  = angle(p['LS'], p['LE'], p['LH'])
        sit   = angle(p['LS'], p['LH'], p['LK'])

        scores = {
            "squat": abs(170 - squat),
            "pushup": abs(170 - push),
            "situp": abs(170 - sit)
        }

        ex = min(scores, key=scores.get)
        STATE["exercise"] = ex

        if ex == "squat":
            analyze(squat, down=155, up=150, good=(85,150))
        elif ex == "pushup":
            analyze(push, down=160, up=150, good=(80,150))
        elif ex == "situp":
            analyze(sit, down=160, up=150, good=(80,150))

        frame = r.plot(conf=False, boxes=False)

    PLAYER["vy"] += 1
    PLAYER["y"] += PLAYER["vy"]
    if PLAYER["y"] >= GROUND:
        PLAYER["y"] = GROUND
        PLAYER["vy"] = 0

    speed = GAME["speed"]
    if GAME["slow"] > 0:
        speed = 2
        GAME["slow"] -= 1

    if random.random() < 0.02:
        OBSTACLES.append(Obstacle(640))

    for o in OBSTACLES[:]:
        o.move(speed)
        o.draw(game)
        if abs(o.x - PLAYER["x"]) < o.r + 25:
            popup("SLIPPED!")
            OBSTACLES.remove(o)

    cv2.circle(game, (PLAYER["x"], int(PLAYER["y"])), 25, (0,200,0), -1)

    for t in TEXTS[:]:
        cv2.putText(game, t["txt"], (200, t["y"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        t["y"] -= 2
        t["life"] -= 1
        if t["life"] <= 0:
            TEXTS.remove(t)

    cv2.putText(game, f"SCORE: {GAME['score']}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.putText(frame, f"EXERCISE: {STATE['exercise'].upper()}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"REPS: {STATE['reps']}",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Game", game)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
