from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        name="ignisalert_v3",
        patience=0,
        save=True,
        plots=True,
        device=0,
        workers=0,
        overlap_mask=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        lr0=0.005,
    )

    print("\n✅ Training complete!")
    print("📁 best.pt saved to: runs/detect/ignisalert_v3/weights/best.pt")