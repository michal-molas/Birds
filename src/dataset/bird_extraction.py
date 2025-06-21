import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torch
from typing import List
import PIL.Image as Image
from .model import ImageData
import cv2

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
faster_rcnn.eval()

transform = T.Compose([T.ToTensor()])

def detect_bird_frcnn(imgs: List[ImageData], threshold: float=0.85) -> List[ImageData | None]:
    """
    Detects and crops a region with birds and resizes it to 224x224

    Parameters:
    imgs - list of images
    threshold - region threshold

    Returns:
    list of cropped images of birds or Nones
    """
    img_tensors = [transform(img_data.img).to(device) for img_data in imgs]
    with torch.no_grad():
        batch_preds = faster_rcnn(img_tensors)

    results = []

    for img_data, preds in zip(imgs, batch_preds):
        img_cv = cv2.cvtColor(np.array(img_data.img), cv2.COLOR_RGB2BGR)

        bird_crop = None
        for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
            if score >= threshold and label == 16:  # 16 is bird in COCO
                if bird_crop is not None:
                    # If there is more than 1 bird, then it might be different species so we skip it
                    bird_crop = None
                    break

                x1, y1, x2, y2 = map(int, box)
                crop = img_cv[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((224, 224))
                bird_crop = crop_pil

        if bird_crop:
            img_data.img = bird_crop
            results.append(img_data)
        else:
            results.append(None)

    return results
