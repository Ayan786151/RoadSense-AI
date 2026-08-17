"""
================================================================================
ROAD SENSE AI - UNIFIED MASTER TRAINING PIPELINE (TRAIN ALL-IN-ONE)
================================================================================
Orchestrates end-to-end training of all RoadSense AI intelligence & vision models:
1. Supervised Machine Learning Risk Predictor (Random Forest & Logistic Regression)
2. Helmet & Rider Compliance Detection Model (YOLO)
3. Red-Light & Multi-Violation AI Model (YOLO)
4. IISc AIM Indian Traffic Fine-Tuning Module (YOLO on BMD-45 / UVH-26)
================================================================================
"""

import sys
import time
import argparse
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def check_gpu_environment():
    """Checks and logs hardware acceleration status."""
    print("=" * 80)
    print("🚦 ROADSENSE AI — MASTER UNIFIED MODEL TRAINING SUITE")
    print("=" * 80)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"✅ Hardware Acceleration : NVIDIA GPU DETECTED")
            print(f"   Device Name           : {device_name}")
            print(f"   Dedicated VRAM        : {vram_gb:.2f} GB")
            print(f"   PyTorch CUDA Build    : {torch.version.cuda}")
        else:
            print("ℹ️ Hardware Acceleration : CPU Mode (Install CUDA PyTorch for faster GPU training)")
    except Exception as e:
        print(f"⚠️ PyTorch Check Warning : {e}")
    print("=" * 80)


def train_ml_risk_model():
    """Stage 1: Train Supervised ML Risk Forecaster."""
    print("\n" + "#" * 80)
    print("STAGE 1 / 4: TRAINING SUPERVISED ML RISK FORECASTING MODEL (50 Zones × 52 Weeks)")
    print("#" * 80)
    
    start_t = time.time()
    from models.train_model import run_model_training_pipeline
    try:
        run_model_training_pipeline()
        elapsed = time.time() - start_t
        print(f"✅ Stage 1 Complete in {elapsed:.1f}s -> Exported: models/best_risk_model.pkl")
        return True
    except Exception as e:
        print(f"❌ Stage 1 Error: {e}")
        return False


def train_helmet_vision_model(epochs: int = 25, batch: int = 16):
    """Stage 2: Train Helmet Compliance Model."""
    print("\n" + "#" * 80)
    print(f"STAGE 2 / 4: TRAINING HELMET & TWO-WHEELER COMPLIANCE MODEL ({epochs} Epochs, Batch {batch})")
    print("#" * 80)
    
    start_t = time.time()
    from vision.train_helmet_model import train_helmet_detector
    try:
        train_helmet_detector(epochs=epochs, batch_size=batch)
        elapsed = time.time() - start_t
        print(f"✅ Stage 2 Complete in {elapsed:.1f}s -> Exported: models/helmet_yolo.pt")
        return True
    except Exception as e:
        print(f"❌ Stage 2 Error: {e}")
        return False


def train_violation_vision_model(epochs: int = 25, batch: int = 16):
    """Stage 3: Train Multi-Violation Detector."""
    print("\n" + "#" * 80)
    print(f"STAGE 3 / 4: TRAINING RED-LIGHT & MULTI-VIOLATION DETECTOR ({epochs} Epochs, Batch {batch})")
    print("#" * 80)
    
    start_t = time.time()
    from vision.train_violation_model import train_violation_model
    try:
        train_violation_model(epochs=epochs, batch_size=batch)
        elapsed = time.time() - start_t
        print(f"✅ Stage 3 Complete in {elapsed:.1f}s -> Exported: models/violation_detector.pt")
        return True
    except Exception as e:
        print(f"❌ Stage 3 Error: {e}")
        return False


def train_iisc_traffic_model(epochs: int = 20, batch: int = 16):
    """Stage 4: Fine-tune on IISc Indian Traffic Benchmark."""
    print("\n" + "#" * 80)
    print(f"STAGE 4 / 4: FINE-TUNING IISc AIM INDIAN TRAFFIC MODEL ({epochs} Epochs, Batch {batch})")
    print("#" * 80)
    
    start_t = time.time()
    from vision.train_uvh26 import train_yolo_uvh26
    try:
        train_yolo_uvh26(epochs=epochs, batch_size=batch)
        elapsed = time.time() - start_t
        print(f"✅ Stage 4 Complete in {elapsed:.1f}s -> Exported: models/iisc_finetuned/")
        return True
    except Exception as e:
        print(f"❌ Stage 4 Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RoadSense AI Unified Master Training Suite")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for YOLO models (default: 25)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--skip-ml", action="store_true", help="Skip Tabular ML Risk Model training")
    parser.add_argument("--skip-vision", action="store_true", help="Skip Vision YOLO models training")
    args = parser.parse_args()

    overall_start = time.time()
    check_gpu_environment()

    results = {}

    # 1. Supervised Risk ML
    if not args.skip_ml:
        results["1. Risk ML Forecaster"] = train_ml_risk_model()
    else:
        results["1. Risk ML Forecaster"] = "SKIPPED"

    # 2-4. Vision Models
    if not args.skip_vision:
        results["2. Helmet Violation YOLO"] = train_helmet_vision_model(epochs=args.epochs, batch=args.batch)
        results["3. Red-Light Violation YOLO"] = train_violation_vision_model(epochs=args.epochs, batch=args.batch)
        results["4. IISc Indian Traffic YOLO"] = train_iisc_traffic_model(epochs=max(10, args.epochs - 5), batch=args.batch)
    else:
        results["2. Helmet Violation YOLO"] = "SKIPPED"
        results["3. Red-Light Violation YOLO"] = "SKIPPED"
        results["4. IISc Indian Traffic YOLO"] = "SKIPPED"

    total_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("🏆 UNIFIED TRAINING SUITE SUMMARY")
    print("=" * 80)
    for model_name, status in results.items():
        symbol = "✅ SUCCESS" if status is True else ("⏩ SKIPPED" if status == "SKIPPED" else "❌ FAILED")
        print(f" • {model_name:<35} : {symbol}")
    print(f"\nTotal Pipeline Elapsed Time: {total_time/60:.2f} minutes")
    print("=" * 80)
    print("Ready to commit & push weights to GitHub via:")
    print("   git add -f models/*.pt models/*.pkl")
    print("   git commit -m 'feat: complete unified training suite weights'")
    print("   git push origin main")
    print("=" * 80)


if __name__ == "__main__":
    main()
