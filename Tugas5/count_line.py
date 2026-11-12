import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO
import supervision as sv

def resolve_path(path: str, fallback_dir: str = "") -> str:
    if os.path.exists(path):
        return path
    if fallback_dir:
        alt_path = os.path.join(fallback_dir, path)
        if os.path.exists(alt_path):
            return alt_path
    parent_alt = os.path.join("..", path)
    if os.path.exists(parent_alt):
        return parent_alt
    return path

def main(source: str, model_path: str, line_y: int):
    source = resolve_path(source, fallback_dir="data")
    model_path = resolve_path(model_path, fallback_dir="Tugas4")

    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video source '{source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[LineZone] Video Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"[LineZone] Menjalankan YOLOv8 pada device: {device} ({device_name})")
    model = YOLO(model_path)

    line_start = sv.Point(x=0, y=line_y)
    line_end = sv.Point(x=width, y=line_y)

    line_zone = sv.LineZone(
        start=line_start,
        end=line_end,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,)
    )

    line_annotator = sv.LineZoneAnnotator(
        thickness=3,
        text_thickness=2,
        text_scale=0.8,
        color=sv.Color.from_hex("#00FFFF"),
        custom_in_text="IN (Masuk): ",
        custom_out_text="OUT (Keluar): "
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    target_classes = ["car", "motorcycle", "bus", "truck", "train"]
    target_ids = [cid for cid, name in model.model.names.items() if name in target_classes]

    class_counts = {cls_name: {"in": 0, "out": 0} for cls_name in ["car", "motorcycle", "bus", "truck"]}

    print("Tekan 'q' pada jendela video untuk berhenti.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai diputar.")
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0 and detections.class_id is not None and detections.tracker_id is not None:
            mask = np.isin(detections.class_id, target_ids) & (detections.tracker_id != None)
            detections = detections[mask]
        else:
            detections = sv.Detections.empty()

        if len(detections) > 0 and detections.tracker_id is not None:
            crossed_in, crossed_out = line_zone.trigger(detections=detections)
        else:
            crossed_in, crossed_out = np.array([]), np.array([])

        if detections.tracker_id is not None:
            for i, (cin, cout) in enumerate(zip(crossed_in, crossed_out)):
                if cin or cout:
                    cls_id = detections.class_id[i]
                    raw_name = model.model.names[cls_id]
                    cls_name = "truck" if raw_name == "train" else raw_name
                    if cls_name in class_counts:
                        if cin:
                            class_counts[cls_name]["in"] += 1
                        if cout:
                            class_counts[cls_name]["out"] += 1

        frame = box_annotator.annotate(scene=frame, detections=detections)
        if detections.tracker_id is not None:
            labels = [
                f"#{tid} {model.model.names[cid]} {conf:.2f}"
                for cid, tid, conf in zip(detections.class_id, detections.tracker_id, detections.confidence)
            ]
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        frame = line_annotator.annotate(frame=frame, line_counter=line_zone)

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, "TRAFFIC ANALYSIS: VEHICLE COUNTING", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Total IN  (Masuk)  : {line_zone.in_count}", (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Total OUT (Keluar) : {line_zone.out_count}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Flow         : {line_zone.in_count + line_zone.out_count} vehicles", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Tugas 5 - Vehicle Counting (LineZone)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*45)
    print("HASIL AKHIR PENGHITUNGAN KENDARAAN (LINEZONE)")
    print("="*45)
    print(f"Total Kendaraan Masuk  (IN)  : {line_zone.in_count}")
    print(f"Total Kendaraan Keluar (OUT) : {line_zone.out_count}")
    print(f"Total Volume Lalu Lintas     : {line_zone.in_count + line_zone.out_count}")
    print("\nRincian per Kelas Kendaraan:")
    for cls_name, cnt in class_counts.items():
        print(f" - {cls_name.capitalize():<12} : IN={cnt['in']}, OUT={cnt['out']}")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tugas 5: Vehicle Counting menggunakan Supervision LineZone")
    parser.add_argument("--source", type=str, default="data/sample_cctv_2.mp4", help="Path ke file video lalu lintas")
    parser.add_argument("--model", type=str, default="Tugas4/yolov8n.pt", help="Path ke model YOLO weights")
    parser.add_argument("--line_y", type=int, default=500, help="Posisi Y garis penghitung (default: 500)")
    args = parser.parse_args()

    main(args.source, args.model, args.line_y)
