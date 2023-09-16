import cv2
import numpy as np
import os

def cv_bbox(image):
    gray = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray, 80, 255, apertureSize=3)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate =cv2.dilate(edges,kernal,iterations = 3)
    lines = cv2.HoughLinesP(dilate, 1, np.pi / 180, 100, minLineLength=500, maxLineGap=10)
    min_x = float('inf')
    min_y = float('inf')
    max_x = 0
    max_y = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
    x1 = min_x
    y1 = min_y
    x2 = max_x
    aspect_ratio = 4 / 5
    h = int((x2 - x1) / aspect_ratio)
    y2 = h+y1 - int(0.01*(h))
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 30)
    return image


image_folder = 'Demo images'
output_folder = 'output_images'
os.makedirs('output_images',exist_ok=True)


for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    result_image = cv_bbox(image)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result_image)
