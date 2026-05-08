from loss_improve import patch_awl
from train_common import run_train


def main():
    # 启用 AWL，只改 loss
    patch_awl(alpha=0.6, verbose=True)

    print("=" * 60)
    print("[Experiment] AWL only")
    print("[Model] baseline YOLO11n")
    print("[Patch] AWL enabled")
    print("[Patch] No CSAM")
    print("[Patch] No P2 head")
    print("=" * 60)

    run_train(
        model_path="yolo11n.pt",
        run_name="B_awl_only",
        epochs=200,
        batch=32,
        imgsz=640,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        workers=0,
        resume=False,
    )


if __name__ == "__main__":
    main()