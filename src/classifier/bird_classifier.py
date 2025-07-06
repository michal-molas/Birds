from tqdm import tqdm
import torch
from torch import nn, optim
from torchvision import models
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from PIL import Image

dataset_name = "michalmolas/birds-of-poland"
batch_size = 64
epochs = 10
val_split = 0.2
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(23),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

hf_dataset = load_dataset(dataset_name)

dataset = hf_dataset["train"]

class_names = dataset.features["label"].names
num_classes = len(class_names)

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

n = len(dataset)
n_val = int(val_split * n)
n_train = n - n_val
hf_train, hf_val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_dataset = HFDataset(hf_train, transform=train_transform)
val_dataset = HFDataset(hf_val, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    tqdm_loop = tqdm(train_loader)

    for imgs, labels in tqdm_loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        tqdm_loop.set_postfix(loss=running_loss / (tqdm_loop.n + 1), acc=train_correct / train_total)

    avg_train_loss = running_loss / len(train_loader)

    model.eval()

    correct = 0
    correct_top2 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            _, top2_preds = outputs.topk(2, dim=1, largest=True, sorted=True)
            correct_top2 += top2_preds.eq(labels.view(-1, 1).expand_as(top2_preds)).any(dim=1).sum().item()

            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).any(dim=1).sum().item()

            _, top10_preds = outputs.topk(10, dim=1, largest=True, sorted=True)
            correct_top10 += top10_preds.eq(labels.view(-1, 1).expand_as(top10_preds)).any(dim=1).sum().item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_acc_top2 = correct_top2 / total
    val_acc_top5 = correct_top5 / total
    val_acc_top10 = correct_top10 / total
    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}", end="")
    print(f" - Top2 Acc: {val_acc_top2:.4f} - Top5 Acc: {val_acc_top5:.4f} - Top10 Acc: {val_acc_top10:.4f}")

torch.save(model.state_dict(), "mobilenetv3_bird_classifier.pt")
