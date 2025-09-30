from django.shortcuts import render
from .utils import load_model, predict_digit
import os
from django.conf import settings


def mnist_inference(request):
    # 固定图片路径
    image_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'test_digit.jpg')

    # 固定模型路径
    model_path = os.path.join(settings.BASE_DIR, 'models', 'mnist_cnn.pt')

    # 加载模型
    model = load_model(model_path)

    # 进行推理
    digit, probabilities = predict_digit(model, image_path)

    # 准备概率数据用于图表显示
    prob_data = [{'digit': i, 'probability': round(prob, 2)} for i, prob in enumerate(probabilities)]

    context = {
        'predicted_digit': digit,
        'probabilities': prob_data,
        'image_url': '/static/images/test_digit.jpg'
    }

    return render(request, 'inference/result.html', context)