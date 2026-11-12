import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO
import supervision as sv

def cross_product_2d(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def get_crossing_direction(line_start: tuple, line_end: tuple, point_before: tuple, point_after: tuple) -> int:
    d_before = cross_product_2d(line_start, line_end, point_before)
    d_after = cross_product_2d(line_start, line_end, point_after)

    if d_before > 0 and d_after < 0:
        return 1   # IN
    elif d_before < 0 and d_after > 0:
        return -1  # OUT
    return 0

class MathLineCrossingCounter:
    def __init__(self, line_start: tuple, line_end: tuple):
        self.line_start = line_start
        self.line_end = line_end
        self.previous_positions = {}
        self.crossed_ids = set()
        self.in_count = 0
        self.out_count = 0

    def update(self, detections) -> dict:
        if detections.tracker_id is None:
            return {"in": self.in_count, "out": self.out_count}

        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            if tracker_id is None:
                continue

            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int(xyxy[3])
            current_pos = (cx, cy)

            if tracker_id in self.previous_positions:
                prev_pos = self.previous_positions[tracker_id]
                direction = get_crossing_direction(self.line_start, self.line_end, prev_pos, current_pos)

                if direction != 0 and tracker_id not in self.crossed_ids:
                    self.crossed_ids.add(tracker_id)
                    if direction == 1:
                        self.in_count += 1
                    else:
                        self.out_count += 1

            d_curr = cross_product_2d(self.line_start, self.line_end, current_pos)
            if d_curr != 0:
                self.previous_positions[tracker_id] = current_pos
            elif tracker_id not in self.previous_positions:
                self.previous_positions[tracker_id] = current_pos

        active_ids = set(tid for tid in detections.tracker_id if tid is not None)
        for old_id in list(self.previous_positions.keys()):
            if old_id not in active_ids:
                del self.previous_positions[old_id]

        return {"in": self.in_count, "out": self.out_count}

def resolve_path(path: str, fallback_dir: str = "") -> str:
    if os.path.exists(path):
        return path
    if fallback_dir:
        alt = os.path.join(fallback_dir, path)
        if os.path.exists(alt): return alt
    alt2 = os.path.join("..", path)
    if os.path.exists(alt2): return alt2
    return path

def main(source: str, model_path: str, line_y: int):
    source = resolve_path(source, "data")
    model_path = resolve_path(model_path, "Tugas4")

    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Error opening source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[MathCounter] Menjalankan YOLOv8 pada device: {device} ({torch.cuda.get_device_name(0) if device==0 else 'CPU'})")
    model = YOLO(model_path)

    counter = MathLineCrossingCounter(line_start=(0, line_y), line_end=(width, line_y))
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    target_classes = ["car", "motorcycle", "bus", "truck", "train"]
    target_ids = [cid for cid, name in model.model.names.items() if name in target_classes]

    print("Tekan 'q' pada jendela video untuk berhenti.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0 and detections.class_id is not None:
            mask = np.isin(detections.class_id, target_ids)
            detections = detections[mask]

        counter.update(detections)

        # Gambar garis manual
        cv2.line(frame, counter.line_start, counter.line_end, (0, 255, 255), 3)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        if detections.tracker_id is not None:
            labels = [f"#{tid} {model.model.names[cid]}" for cid, tid in zip(detections.class_id, detections.tracker_id)]
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # HUD Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (420, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "MATH CROSS PRODUCT 2D COUNTER", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"IN (Left to Right)  : {counter.in_count}", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT (Right to Left) : {counter.out_count}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Crossed       : {counter.in_count + counter.out_count} vehicles", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Tugas 5 - Math Cross Product Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*45)
    print("HASIL AKHIR (MATH CROSS PRODUCT COUNTER)")
    print("="*45)
    print(f"Total IN  : {counter.in_count}")
    print(f"Total OUT : {counter.out_count}")
    print(f"Total     : {counter.in_count + counter.out_count}")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tugas 5: Vehicle Counting menggunakan Matematika Cross Product 2D")
    parser.add_argument("--source", type=str, default="data/sample_cctv_2.mp4", help="Path ke file video lalu lintas")
    parser.add_argument("--model", type=str, default="Tugas4/yolov8n.pt", help="Path ke model YOLO weights")
    parser.add_argument("--line_y", type=int, default=500, help="Posisi Y garis penghitung (default: 500)")
    args = parser.parse_args()

    main(args.source, args.model, args.line_y)
