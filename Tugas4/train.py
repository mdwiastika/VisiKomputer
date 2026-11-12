import os
import torch
from ultralytics import YOLO
import argparse

def train_model(data_config, epochs, device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "CPU"
    print(f"Starting training on device: {device} ({device_name})")

    # Resolve data config path if relative
    if not os.path.isabs(data_config) and not os.path.exists(data_config):
        alt_path = os.path.join("..", data_config)
        if os.path.exists(alt_path):
            data_config = alt_path

    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_config, 
        epochs=epochs, 
        imgsz=640,
        plots=True,
        device=device,
        project="../models",
        name="traffic_custom"
    )

    print("Training complete.")
    print(f"Best model saved to: {results.save_dir}")

    # Export the model
    try:
        success = model.export(format="onnx")
        print(f"Model exported to ONNX: {success}")
    except Exception as e:
        print(f"ONNX export notice: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 02: Custom Model Training")
    parser.add_argument("--data", type=str, default="datasets/traffic/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device to use: '0' for GPU or 'cpu' (default auto)")
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.device)
