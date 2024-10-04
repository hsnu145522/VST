from model import ClassificationModel

import torch
import torch.nn as nn
from lion_pytorch import Lion
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image
import os

image_list = []
label_dict = {}

def LoadDataset(train_dir):
    categories = os.listdir(train_dir)
    categories.sort()
    
    count = 0
    for category in categories:
        label_dict[category] = count
        count+=1

        newpath = os.path.join(train_dir, category)
        image_path = os.listdir(newpath)
        for image in image_path:
            newimage = os.path.join(category, image)
            image_list.append(newimage)
    image_list.sort()

class CustomDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.data = image_list
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]
        img_name = os.path.join(self.images_dir, name) 
        category_name = name[:name.find("/")]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        label = label_dict[category_name]

        return image, label

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device):
    best_loss = float('inf')
    best_accuracy = 0
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                maxk = max((1,5))
                _, predicted = outputs.topk(maxk, 1, True, True)
                # _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                labels_resize = labels.view(-1,1)
                correct += torch.eq(predicted, labels_resize).sum().float().item()

        val_loss = val_loss / len(val_loader)
        accuracy = 100 * (correct / total)

        print(f'Epoch {epoch + 1} | Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Top-5 Acc: {accuracy:.2f}%')

        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = accuracy
            best_model_wts = model.state_dict()

    return best_loss, best_accuracy, best_model_wts

def train():
    images_dir = "./dataset/train"
    LoadDataset(images_dir)
    dataset = CustomDataset(images_dir)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_loss_overall = float('inf')
    best_accuracy_overall = 0
    best_model_wts_overall = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Training fold {fold + 1}')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        model = ClassificationModel()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)

        best_loss, best_accuracy, best_model_wts = train_one_fold(model, train_loader, val_loader, criterion, optimizer, device)
        print(f'Best validation loss for fold {fold + 1}: {best_loss:.4f}, Top-5 Accuracy: {best_accuracy:.2f}%')

        if best_loss < best_loss_overall:
            best_loss_overall = best_loss
            best_accuracy_overall = best_accuracy
            best_model_wts_overall = best_model_wts
            model.load_state_dict(best_model_wts_overall)
            torch.save(model.state_dict(), f'w_313581001_fold{fold + 1}_smaller.pth')

    print(f'Best validation loss overall: {best_loss_overall:.4f}, TOp-5 Accuracy: {best_accuracy_overall:.2f}%')
    model.load_state_dict(best_model_wts_overall)
    torch.save(model.state_dict(), 'w_313581001_smaller.pth')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    train()
    
