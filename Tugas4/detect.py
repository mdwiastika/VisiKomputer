import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import supervision as sv

def main(source, model_path):
    # Resolve paths if run from root or Tugas4
    if not os.path.exists(model_path) and os.path.exists(os.path.join("Tugas4", model_path)):
        model_path = os.path.join("Tugas4", model_path)

    if not source.isdigit() and not os.path.exists(source):
        alt_source = os.path.join("data", os.path.basename(source))
        if os.path.exists(alt_source):
            source = alt_source

    model = YOLO(model_path)

    cap = cv2.VideoCapture(source) if not str(source).isdigit() else cv2.VideoCapture(int(source))
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Daftar kelas kendaraan lalu lintas yang ingin dideteksi
    target_classes = ["car", "motorcycle", "bus", "truck"]
    target_ids = [cid for cid, name in model.model.names.items() if name in target_classes]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Inference
        results = model(frame, verbose=False)[0]

        # 2. Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # 3. Filter: hanya tampilkan kendaraan (mobil, motor, bus, truk)
        # Menghilangkan orang jalan kaki (person), tas (backpack), dsb.
        if len(target_ids) > 0 and len(detections) > 0:
            detections = detections[np.isin(detections.class_id, target_ids)]

        # 4. Annotate
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            labels=[f"{model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        )

        cv2.imshow("Lab 03 - Object Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 03: Object Detection")
    parser.add_argument("--source", type=str, default="data/sample.mp4", help="Path to video file")
    parser.add_argument("--model", type=str, default="Tugas4/yolov8n.pt", help="Path to model weights")
    args = parser.parse_args()

    main(args.source, args.model)

