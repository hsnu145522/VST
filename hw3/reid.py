import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

class ReIDFeatureExtractor:
    def __init__(self, model_path=None):
        self.model = models.resnet50(pretrained=True)  # Use a lightweight CNN for simplicity
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def extract(self, img):
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0)  # Prepare image for the model
        with torch.no_grad():
            feature = self.model(img).cpu().numpy()
        return feature / np.linalg.norm(feature)  # Normalize the feature vector
