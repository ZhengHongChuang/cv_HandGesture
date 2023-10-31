'''
Descripttion: Leetcode_code
version: 1.0
Author: zhc
Date: 2023-10-28 21:07:56
LastEditors: zhc
LastEditTime: 2023-10-28 21:42:19
'''
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
def classify_image(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == '__main__':
    model_path = "weight/model_100.pth"
    image_path = "run/one/gesture-one-2021-03-07_23-07-48-24_61901.jpg"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 14),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    predicted_class = classify_image(model, image_path, device)

    print(f"The image is classified as class {predicted_class}")
