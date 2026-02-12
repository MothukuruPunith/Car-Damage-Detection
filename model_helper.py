import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trained_model = None

class_names = [
    'F_Breakage',
    'F_Crushed',
    'F_Normal',
    'R_Breakage',
    'R_Crushed',
    'R_Normal'
]

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet(6)
        trained_model.load_state_dict(
            torch.load("saved_model.pth", map_location=device)
        )
        trained_model.to(device)
        trained_model.eval()

    return trained_model


def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    model = load_model()

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]
