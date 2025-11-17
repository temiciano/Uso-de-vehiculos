import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("./video/trozo648.mp4")
cv2.namedWindow("Detección de Autos", cv2.WINDOW_NORMAL)

roi_area2 = np.array([
    (109,186),
    (147,260),
    (227,248),
    (180,178)
],np.int32)
#Puerta garage 0 y 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.polylines(frame, [roi_area2], isClosed=True, color=(255, 0, 0), thickness=1)

    results = model.predict(source=frame, verbose=False)
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        inside = cv2.pointPolygonTest(roi_area2, (cx, cy), False)

        if inside >= 0:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.polylines(annotated_frame, [roi_area2], isClosed=True, color=(255, 0, 0), thickness=1)

    cv2.imshow("Detección de Autos", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()