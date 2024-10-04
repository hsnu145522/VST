from model import ClassificationModel
from utils import Label_Map

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import csv

image_list = []

def LoadDataset(test_dir):
    images = os.listdir(test_dir)
    for image in images:
        image_list.append(image)
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
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        return image, name

def test():
    test_dir = "./dataset/test"
    weight_path = "./w_313581001.pth"
    LoadDataset(test_dir)
    dataset = CustomDataset(test_dir)

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device:{device}")

    model = ClassificationModel()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    with open('pred_313581001.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5'])

        with torch.no_grad():
            for images, filename in test_loader:
                images = images.to(device)
                outputs = model(images)
                maxk = max((1,5))
                _, predicted = outputs.topk(maxk, 1, True, True)

                pred_list = predicted.tolist()
                pred_list = [Label_Map[i] for i in pred_list[0]]
                writer.writerow([filename[0], *pred_list])
                

if __name__ == "__main__":
    test()
