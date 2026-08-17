"""
================================================================================
ROAD SENSE AI - VIOLATION DETECTION MODEL TRAINING MODULE
MODULE: HELMET / NO-HELMET & TRAFFIC LIGHT FINE-TUNING PIPELINE
================================================================================

This module provides training and fine-tuning pipelines using Ultralytics YOLO
for safety violation detection:
1. Helmet / No-Helmet Detection (Classes: helmet, no_helmet, rider, head)
2. Traffic Light State Classification (Classes: red_light, yellow_light, green_light)

Supported dataset formats:
- Roboflow export YAML
- Custom annotated YOLO dataset YAML (data/safety_dataset.yaml)
- IISc AIM / OpenImages safety subsets
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


# ==============================================================================
# DEFAULT CLASS SCHEMAS
# ==============================================================================

HELMET_CLASSES = [
    "helmet",
    "no_helmet",
    "rider",
    "motorcycle"
]

TRAFFIC_LIGHT_CLASSES = [
    "red_light",
    "yellow_light",
    "green_light",
    "off"
]


# ==============================================================================
# SAMPLE DATASET YAML GENERATOR (FOR LOCAL QUICK-START)
# ==============================================================================

def create_sample_dataset_yaml(task_type: str = "helmet", output_path: Optional[str] = None) -> str:
    """
    Creates a template dataset YAML configuration for helmet or traffic light fine-tuning.
    """
    if output_path is None:
        output_path = f"data/{task_type}_dataset.yaml"

    if task_type == "traffic_light":
        dataset_dir = Path("datasets/traffic_light_dataset").resolve().as_posix()
        classes_str = "  0: red_light\n  1: yellow_light\n  2: green_light\n  3: off"
    else:
        dataset_dir = Path("datasets/helmet_dataset").resolve().as_posix()
        classes_str = "  0: helmet\n  1: no_helmet\n  2: rider\n  3: motorcycle"

    yaml_content = f"""# RoadSense AI - {task_type.title()} Detection Dataset Config
path: {dataset_dir}
train: images/train
val: images/val

names:
{classes_str}
"""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(yaml_content)
    print(f"[+] Created {task_type} dataset config: {output_path}")
    return str(p)


# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train_violation_detector(
    task_type: str = "helmet",
    data_yaml: Optional[str] = None,
    base_model: str = "yolo11n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    img_size: int = 640,
    output_dir: str = "models/violation_models",
    device: str = "cpu"
):
    """
    Fine-tunes a base YOLO model for helmet or traffic light violation detection.
    """
    if data_yaml is None:
        data_yaml = f"data/{task_type}_dataset.yaml"

    print("=" * 70)
    print(f"ROAD SENSE AI - {task_type.upper()} VIOLATION MODEL TRAINING")
    print("=" * 70)
    print(f"Task Type     : {task_type}")
    print(f"Base Model    : {base_model}")
    print(f"Dataset YAML  : {data_yaml}")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Image Size    : {img_size}")
    print(f"Target Output : {output_dir}")
    print(f"Device        : {device}")
    print("=" * 70)

    # 1. Check/create dataset config if needed
    if not Path(data_yaml).exists():
        print(f"[!] Dataset config '{data_yaml}' not found. Generating template...")
        data_yaml = create_sample_dataset_yaml(task_type=task_type, output_path=data_yaml)

    # 2. Load model
    print(f"\n[+] Loading base model: {base_model}...")
    model = YOLO(base_model)

    run_name = f"{task_type}_detector"

    # 3. Train
    print("\n[+] Starting model training / fine-tuning...")
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=output_dir,
            name=run_name,
            save=True,
            exist_ok=True,
            device=device if device != "auto" else None
        )
        print("\n" + "=" * 70)
        print(f"[+] {task_type.upper()} TRAINING COMPLETE!")
        print(f"[+] Weights saved to: {output_dir}/{run_name}/weights/best.pt")
        print("=" * 70)
        return results
    except Exception as e:
        print(f"\n[!] Training halted or mock executed: {e}")
        print(f"[i] To train with custom data, place annotated images in ./datasets/{task_type}/")
        return None


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI - Train Helmet & Traffic Light Violation Models")
    parser.add_argument("--task", type=str, default="helmet", choices=["helmet", "traffic_light"], help="Target detection task")
    parser.add_argument("--data", type=str, default="data/helmet_dataset.yaml", help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights (e.g. yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--output", type=str, default="models/violation_models", help="Directory to save trained weights")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', 'cuda')")

    args = parser.parse_args()

    train_violation_detector(
        task_type=args.task,
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        output_dir=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
