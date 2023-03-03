import cv2
import numpy as np
import os
from joblib import Parallel, delayed

def denoise(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return dst

def batch_denoise(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = os.listdir(input_folder)
    results = Parallel(n_jobs=-1)(delayed(process_file)(input_folder, output_folder, file) for file in files)

def process_file(input_folder, output_folder, file):
    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)
    image = cv2.imread(input_path)
    denoised = denoise(image)
    cv2.imwrite(output_path, denoised)

# Example usage
input_folder = 'input_images/'
output_folder = 'output_images/'
batch_denoise(input_folder, output_folder)
