import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 类别
class_names = ['paper', 'rock', 'scissors']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # 使用最新的weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()

# 图像预处理
def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3]),  # 只保留前三个通道（RGB）
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 格式
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

# 预测函数
def predict(image_path):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

# 示例：预测单张图像的类别
image_path = '/Users/xiaofuqing/Desktop/石头剪刀布/original_data/paper/paper01-000.png'
predicted_class = predict(image_path)
print({predicted_class})