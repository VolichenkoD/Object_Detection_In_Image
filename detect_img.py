import torchvision
import torch
import argparse
import cv2
import detect_utils

from PIL import Image

# собираем парсер аргументов
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
args = vars(parser.parse_args())

# определяем вычислительное устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# загружаем модель 
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# загружаем модель на вычислительное устройство
model.eval().to(device)

# читаем изображение и запускаем вывод для обнаружения
image = Image.open(args['input'])
boxes, classes, labels = detect_utils.predict(image, model, device, 0.7)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
# cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
# cv2.waitKey(0)
