import os
import cv2
import numpy as np
import argparse
import torch
from ultralytics import YOLO
import supervision as sv

def resolve_path(path: str, fallback_dir: str = "") -> str:
    """Helper untuk memastikan path file valid jika dipanggil dari root atau folder Tugas5."""
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

def main(source: str, model_path: str):
    # Resolusi path
    source = resolve_path(source, fallback_dir="data")
    model_path = resolve_path(model_path, fallback_dir="Tugas4")

    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video source '{source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[PolygonZone] Video Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    # Perangkat akselerasi GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"[PolygonZone] Menjalankan YOLOv8 pada device: {device} ({device_name})")
    model = YOLO(model_path)

    # Definisikan Area Poligon (Zona Pantau Jalan Raya)
    # Bentuk trapesium perspektif jalan: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    polygon = np.array([
        [int(width * 0.20), int(height * 0.40)],
        [int(width * 0.80), int(height * 0.40)],
        [int(width * 0.95), int(height * 0.92)],
        [int(width * 0.05), int(height * 0.92)]
    ], dtype=np.int32)

    # Inisialisasi PolygonZone
    zone = sv.PolygonZone(
        polygon=polygon,
        triggering_anchors=(sv.Position.BOTTOM_CENTER,)
    )

    # Annotators
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.from_hex("#00FF00"),
        thickness=2
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Filter kelas kendaraan
    target_classes = ["car", "motorcycle", "bus", "truck"]
    target_ids = [cid for cid, name in model.model.names.items() if name in target_classes]

    print("Tekan 'q' pada jendela video untuk berhenti.\n")

    max_occupancy = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai diputar.")
            break

        # 1. Inference
        results = model.track(frame, persist=True, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. Filter kelas kendaraan
        if len(detections) > 0 and detections.class_id is not None:
            mask = np.isin(detections.class_id, target_ids)
            detections = detections[mask]

        # 3. Trigger PolygonZone
        is_in_zone = zone.trigger(detections=detections)
        vehicles_in_zone = int(np.sum(is_in_zone))
        if vehicles_in_zone > max_occupancy:
            max_occupancy = vehicles_in_zone

        # 4. Gambar Visualisasi
        frame = zone_annotator.annotate(scene=frame)
        frame = box_annotator.annotate(scene=frame, detections=detections)

        if detections.tracker_id is not None:
            labels = [
                f"#{tid} {model.model.names[cid]}"
                for cid, tid in zip(detections.class_id, detections.tracker_id)
            ]
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # 5. HUD Status Kepadatan Zona
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (420, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Ubah warna indikator kepadatan
        status_color = (0, 255, 0) if vehicles_in_zone < 4 else (0, 165, 255) if vehicles_in_zone < 8 else (0, 0, 255)
        status_text = "LANCAR" if vehicles_in_zone < 4 else "PADAT" if vehicles_in_zone < 8 else "MACET / ANTRIAN"

        cv2.putText(frame, "POLYGON ZONE: AREA OCCUPANCY", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Kendaraan di Zona : {vehicles_in_zone}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Status Lalu Lintas : {status_text}", (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("Tugas 5 - Area Counting (PolygonZone)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*45)
    print("HASIL AKHIR MONITORING ZONA (POLYGONZONE)")
    print("="*45)
    print(f"Puncak Kepadatan Zona : {max_occupancy} kendaraan secara bersamaan")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tugas 5: Area Counting menggunakan Supervision PolygonZone")
    parser.add_argument("--source", type=str, default="data/sample_cctv_2.mp4", help="Path ke file video lalu lintas")
    parser.add_argument("--model", type=str, default="Tugas4/yolov8n.pt", help="Path ke model YOLO weights")
    args = parser.parse_args()

    main(args.source, args.model)
