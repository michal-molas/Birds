import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large
from ultralytics import YOLO
from PIL import Image
import time

LABELS_PATH = "data/class_labels.txt"
CLASSIFIER_PATH = "mobilenetv3_bird_classifier.pt"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BIRD_CLASS_ID = 14
INFERENCE_INTERVAL = 0.2

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

def detect_bird_bboxes(frame, yolo_model):
    results = yolo_model(frame)[0]
    bird_boxes = [box.xyxy[0].int().tolist() for box in results.boxes if int(box.cls) == BIRD_CLASS_ID]
    
    return bird_boxes

def main():
    idx_to_class = load_labels()
    model = load_mobilenet(len(idx_to_class))
    yolo_model = YOLO("yolo11n.pt") # YOLO("yolov8n.pt")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    last_inference_time = 0
    predictions = []
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_inference_time > INFERENCE_INTERVAL:
            predictions = []
            bboxes = detect_bird_bboxes(frame, yolo_model)

            for box in bboxes:
                x1, y1, x2, y2 = box
                cropped_img = frame[y1:y2, x1:x2]

                img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(img_tensor)

                _, preds = torch.max(outputs, 1)
                predicted_idx = preds.item()
                predicted_class = idx_to_class[predicted_idx]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Bird Detector", frame)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break


            # x1, y1, x2, y2 = box
            # cropped_images.append(frame[y1:y2, x1:x2])

        #     results = yolo_model(frame)[0]
        #     for box in results.boxes:
        #         cls = int(box.cls.item())
        #         if cls == 14:  # COCO class 14 = bird
        #             x1, y1, x2, y2 = map(int, box.xyxy[0])
        #             cropped = frame[y1:y2, x1:x2]
        #             if cropped.size == 0:
        #                 continue
        #             label = classify_crop(cropped, model, labels)
        #             predictions.append((x1, y1, x2, y2, label))
        #     last_inference_time = current_time

        # for x1, y1, x2, y2, label in predictions:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, label, (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # cv2.imshow("Bird Detector", frame)
        # if cv2.waitKey(1) == 27:  # ESC to quit
        #     break

    cap.release()

if __name__ == "__main__":
    main()
