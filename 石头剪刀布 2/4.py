import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# 类别名称
class_names = ['paper', 'rock', 'scissors']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)  # 使用最新的 'weights' 参数而不是 'pretrained'
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()


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


def predict(image_path):
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]


# 遍历文件夹中的所有图像并进行预测
def predict_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):  # 根据需要检查其他图像扩展名
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict(image_path)
            results.append((filename, predicted_class))
            print(f'Image: {filename}, Predicted class: {predicted_class}')
    return results


# 示例：预测整个文件夹中的图像类别
folder_path = ' '  # 替换为图像文件夹路径
results = predict_folder(folder_path)

with open('predictions.txt', 'w') as f:  # 可以更换保存文件夹
    for filename, predicted_class in results:
        f.write(f'Image: {filename}, Predicted class: {predicted_class}\n')
