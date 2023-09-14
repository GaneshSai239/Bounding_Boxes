import cv2
import mediapipe as mp
import numpy as np
import os


def segmentation(image):
    SelfieSegmentation = mp.solutions.selfie_segmentation.SelfieSegmentation
    selfie_segmentation = SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask =results.segmentation_mask> 0.3
    return mask

def bounding_box(mask,image):
    
    _, threshold = cv2.threshold((mask*255).astype(np.uint8),128,255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)  # Largest contour
    x,y,w,h = cv2.boundingRect(largest_contour)
    y = int(y - (h*0.04))
    h = int(h+(h*0.04))
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),30)
    return image

image_folder = 'Demo images'
output_folder = 'output_images'
os.makedirs('output_images',exist_ok=True)


for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    results = segmentation(image)
    result_image = bounding_box(results,image)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result_image)
