from ultralytics import YOLO
from loss_improve import patch_awl

DATA_YAML = r"coda.yaml"
PROJECT_DIR = r"runs"

if __name__ == "__main__":
    # 开启 AWL，6:4
    patch_awl(alpha=0.6)

    model = YOLO(r"D:\Project\yolo11\yolo11_ema_cisa_p2.yaml")

    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=640,
        batch=32,
        device=0,
        project=PROJECT_DIR,
        name="G_ours",
        optimizer="SGD",
        lr0=0.01,
        workers=0,
        resume=False,
    )