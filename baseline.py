from train_common import run_train

if __name__ == "__main__":
    run_train(
        model_path=r"D:\Project\yolo11\yolo11n.pt",
        run_name="A_baseline",
        epochs=200,
        batch=32,
        imgsz=640,
        device=0,
        optimizer="SGD",
        lr0=0.01,
        workers=0,
    )