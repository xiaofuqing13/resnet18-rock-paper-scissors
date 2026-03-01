import os
import shutil
from sklearn.model_selection import train_test_split

# 定义数据集路径
original_data_dir = 'original_data'
base_dir = 'dataset'
os.makedirs(base_dir, exist_ok=True)

# 定义训练、验证和测试集目录
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# 获取类别
categories = ['paper', 'rock', 'scissors']

# 按类别创建子目录
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# 划分数据
for category in categories:
    category_dir = os.path.join(original_data_dir, category)
    images = os.listdir(category_dir)
    train_images, temp_images = train_test_split(images, test_size=0.4, random_state=42)
    validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    for image in train_images:
        shutil.copy(os.path.join(category_dir, image), os.path.join(train_dir, category, image))

    for image in validation_images:
        shutil.copy(os.path.join(category_dir, image), os.path.join(validation_dir, category, image))

    for image in test_images:
        shutil.copy(os.path.join(category_dir, image), os.path.join(test_dir, category, image))