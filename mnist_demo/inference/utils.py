import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class MNISTCNN(nn.Module):
    """简单的 CNN 模型用于 MNIST 分类"""

    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path):
    """加载预训练模型"""
    model = MNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(image_path):
    """预处理图片为模型输入格式"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # 添加 batch 维度


def predict_digit(model, image_path):
    """使用模型预测图片中的数字"""
    # 预处理图片
    input_tensor = preprocess_image(image_path)

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100

    return predicted.item(), probabilities.tolist()