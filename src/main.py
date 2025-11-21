import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import csv
import time
import torch

# =====================================================
# CONFIGURACIÓN
# =====================================================
VIDEO_PATH = "./video/output.mp4"
MODEL_PATH = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"    # <-- tracker integrado
OUTPUT_CSV = "conteo_autos.csv"

SKIP_FRAMES = 2            # procesa 1 de cada N
MAX_WIDTH = 960
CONFIDENCE = 0.35

HORA_INICIO = datetime(2025, 11, 10, 6, 0, 0)

ROI_ORIGINAL = np.array([
    (41,158),
    (66,189),
    (104,186),
    (138,260),
    (234,248),
    (187,181),
    (201,141)
], np.int32)

LINE_P1 = (98,186)
LINE_P2 = (183,176)

# =====================================================
# FUNCIONES
# =====================================================
def scale_points(points, scale):
    return np.array([(int(x*scale), int(y*scale)) for (x,y) in points], np.int32)

def crossed_line(prev, curr, p1, p2):
    def side(pt):
        return np.sign((p2[0]-p1[0])*(pt[1]-p1[1]) -
                       (p2[1]-p1[1])*(pt[0]-p1[0]))
    return side(prev), side(curr)

def timestamp_real(frame_idx, fps):
    segundos = frame_idx / fps
    return HORA_INICIO + timedelta(seconds=segundos)

# =====================================================
# CARGAR MODELO + TRACKER
# =====================================================
model = YOLO(MODEL_PATH)

if torch.cuda.is_available():
    try:
        model.to("cuda")
        print("Modelo en CUDA")
    except:
        pass

# =====================================================
# VIDEO
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps_video, "Frames:", total_frames)

# =====================================================
# CSV
# =====================================================
csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow([
    "frame_idx", "timestamp", "evento",
    "track_id", "clase",
    "cx", "cy", "x1", "y1", "x2", "y2"
])

# =====================================================
# VARIABLES
# =====================================================
frame_idx = -1
entradas = 0
salidas = 0
prev_positions = {}   # track_id -> (cx, cy)

start = time.time()
processed = 0

# =====================================================
# LOOP PRINCIPAL
# =====================================================
while True:
    frame_idx += 1
    ret, frame = cap.read()
    if not ret:
        break

    # SKIP FRAMES
    if frame_idx % SKIP_FRAMES != 0:
        continue

    processed += 1

    # Resize
    h, w = frame.shape[:2]
    scale = 1.0
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

    roi_area = scale_points(ROI_ORIGINAL, scale)
    line_p1 = scale_points([LINE_P1], scale)[0]
    line_p2 = scale_points([LINE_P2], scale)[0]

    # =====================================================
    # SE USA EL TRACKER INTEGRADO (BYTETrack)
    # =====================================================
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE,
        tracker=TRACKER_CONFIG,
        verbose=False
    )

    if results[0].boxes.id is None:
        continue

    ids = results[0].boxes.id.cpu().numpy().astype(int)
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = map(int, xyxy[i])
        cls_name = model.names[cls_ids[i]]

        if cls_name not in ["car", "truck", "bus"]:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        inside = cv2.pointPolygonTest(roi_area, (cx, cy), False)
        if inside < 0:
            continue

        curr = (cx, cy)
        prev = prev_positions.get(track_id, curr)

        hora_real = timestamp_real(frame_idx, fps_video).strftime("%Y-%m-%d %H:%M:%S")

        # DETECCIÓN DE CRUCE
        side_p, side_c = crossed_line(prev, curr, line_p1, line_p2)

        if side_p != side_c:
            if curr[1] < prev[1]:
                salidas += 1
                evento = "salida"
            else:
                entradas += 1
                evento = "entrada"

            writer.writerow([
                frame_idx, hora_real, evento,
                track_id, cls_name,
                cx, cy, x1, y1, x2, y2
            ])

        prev_positions[track_id] = curr

# =====================================================
# FIN
# =====================================================
elapsed = time.time() - start

print("\n=== RESULTADOS ===")
print("Frames procesados:", processed)
print("Tiempo total:", round(elapsed, 2), "s")
print("Velocidad:", round(processed/elapsed, 2), "FPS")
print("Entradas:", entradas)
print("Salidas:", salidas)

csv_file.close()
cap.release()
print("\nCSV guardado en", OUTPUT_CSV)
