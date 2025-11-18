import cv2
import numpy as np
from ultralytics import YOLO
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("./video/trozo942.mp4")
cv2.namedWindow("Detección de Autos", cv2.WINDOW_NORMAL)

roi_area2 = np.array([
    (41,158),
    (66,189),
    (104,186),
    (138,260),
    (234,248),
    (187,181),
    (201,141)
],np.int32)
#Puerta garage 0 y 3

line_p1 = (98,186)
line_p2 = (183,176)

entradas = 0
salidas = 0

next_id = 0
tracks = {}  # id -> {cx, cy, prev, class, missed}

def crossed_line(prev, curr, p1, p2):
    def side(pt):
        return np.sign((p2[0]-p1[0])*(pt[1]-p1[1]) -
                       (p2[1]-p1[1])*(pt[0]-p1[0]))
    return side(prev), side(curr)

def match_track(cx, cy, tracks, max_dist=60):
    best_id = None
    best_dist = max_dist

    for tid, t in tracks.items():
        dist = math.hypot(cx - t["cx"], cy - t["cy"])
        if dist < best_dist:
            best_dist = dist
            best_id = tid

    return best_id


while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.polylines(frame, [roi_area2], True, (255,0,0), 1)
    cv2.line(frame, line_p1, line_p2, (0,255,255), 1,4)

    results = model.predict(frame, verbose=False)
    annotated = frame.copy()

    detections = []

    # YOLO detections → lista simple
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        original = model.names[cls_id]

        if original not in ["car", "truck", "bus"]:
            continue

        cls_name = "vehicule"

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        inside = cv2.pointPolygonTest(roi_area2, (cx,cy), False)
        if inside < 0:
            continue

        detections.append((cx, cy, x1, y1, x2, y2, cls_name))

    # ---- TRACKING ----
    used_ids = set()

    for (cx, cy, x1, y1, x2, y2, cls_name) in detections:
        assigned_id = match_track(cx, cy, tracks)

        if assigned_id is None:
            # crear track nuevo
            tracks[next_id] = {
                "cx": cx, "cy": cy,
                "prev": (cx, cy),
                "class": cls_name,  # CLASE FIJA
                "missed": 0
            }
            assigned_id = next_id
            next_id += 1
        else:
            tracks[assigned_id]["cx"] = cx
            tracks[assigned_id]["cy"] = cy
            tracks[assigned_id]["missed"] = 0

        used_ids.add(assigned_id)

        # cruces
        prev = tracks[assigned_id]["prev"]
        curr = (cx, cy)
        side_p, side_c = crossed_line(prev, curr, line_p1, line_p2)

        if side_p != side_c:
            # dirección según movimiento vertical
            if curr[1] < prev[1]:
                salidas += 1
            else:
                entradas += 1

        tracks[assigned_id]["prev"] = curr

        # dibujar
        fixed_class = tracks[assigned_id]["class"]  # CLASE CONGELADA
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(annotated, f"{fixed_class} ID:{assigned_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)

    # eliminar tracks no actualizados
    to_delete = []
    for tid in tracks:
        if tid not in used_ids:
            tracks[tid]["missed"] += 1
            if tracks[tid]["missed"] > 10:
                to_delete.append(tid)

    for tid in to_delete:
        del tracks[tid]

    cv2.putText(annotated, f"Entradas: {entradas}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(annotated, f"Salidas: {salidas}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Detección de Autos", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()