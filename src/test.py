import os
import csv
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# ----------------------
# 1 加载模型
# ----------------------

model = resnet50(weights='DEFAULT')
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('model01.pth', map_location='cpu', weights_only=True))

model.eval()

# ----------------------
# 2 图片预处理
# ----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------
# 3 弹窗选择图片
# ----------------------

Tk().withdraw()  # 不显示主窗口

image_paths = askopenfilenames(
    title="选择要检测的图片",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

# ----------------------
# 4 推理
# ----------------------

results = []

with torch.no_grad():

    for img_path in image_paths:

        try:

            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)

            outputs = model(img)

            # 转概率
            probs = torch.softmax(outputs, dim=1)

            real_prob = probs[0][0].item()
            ai_prob = probs[0][1].item()

            img_name = os.path.basename(img_path)

            print(f"{img_name}  AI概率: {ai_prob:.4f}  真实概率: {real_prob:.4f}")

            results.append((img_name, ai_prob, real_prob))

        except Exception as e:
            print(f"处理失败 {img_path}: {e}")

# ----------------------
# 5 保存CSV
# ----------------------

with open('cla_pre.csv', 'w', newline='') as f:

    writer = csv.writer(f)

    writer.writerow(["image", "ai_prob", "real_prob"])

    writer.writerows(results)

print("\n结果已保存到 cla_pre.csv")