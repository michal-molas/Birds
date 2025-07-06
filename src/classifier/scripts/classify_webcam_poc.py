import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from ultralytics import YOLO
from PIL import Image

LABELS_PATH = "data/class_labels.txt"
CLASSIFIER_PATH = "mobilenetv3_bird_classifier.pt"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_mobilenet(n_classes):
    model = mobilenet_v3_large()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, n_classes)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def detect_bird_crop(frame, yolo_model):
    results = yolo_model(frame)[0]
    BIRD_CLASS_ID = 14
    bird_boxes = [box.xyxy[0].int().tolist() for box in results.boxes if int(box.cls) == BIRD_CLASS_ID]
    
    cropped_images = []

    for box in bird_boxes:
        x1, y1, x2, y2 = box
        cropped_images.append(frame[y1:y2, x1:x2])
    return cropped_images

def capture_and_crop(yolo_model):
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Webcam - press SPACE to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    cropped = detect_bird_crop(frame, yolo_model)

    tensors = []

    for i, img in enumerate(cropped):
        out_filename = f"cropped{i}.jpg"
        cv2.imwrite(out_filename, img)
        print(f"Cropped bird saved to {out_filename}")

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensors.append(transform(img_pil).unsqueeze(0).to(DEVICE))
    return tensors

def main():
    idx_to_class = load_labels()
    model = load_mobilenet(len(idx_to_class))
    yolo_model = YOLO("yolo11n.pt") # YOLO("yolov8n.pt")

    input_tensors = capture_and_crop(yolo_model)

    with torch.no_grad():
        for i, tensor in enumerate(input_tensors):
            outputs = model(tensor)
            _, preds = torch.max(outputs, 1)
            predicted_idx = preds.item()
            predicted_class = idx_to_class[predicted_idx] if idx_to_class else predicted_idx
            print(f"Prediction {i}: {predicted_class}")

if __name__ == "__main__":
    main()
