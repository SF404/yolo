from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = YOLO('yolov8s.pt').to(device)

    data_path = 'C:\\Users\\suhai\\Desktop\\Research\\Models\\Yolo\\data.yaml'
    epochs = 5
    imgsz = 640
    batch_size = 16

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device
    )

if __name__ == '__main__':
    main()
