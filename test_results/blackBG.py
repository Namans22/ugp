import cv2
import mediapipe as mp
import numpy as np

# Load the image
image = cv2.imread('normal.png')  # replace with your image filename

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the segmentation mask
results = segment.process(rgb_image)
mask = results.segmentation_mask

# Threshold the mask to keep person and remove background
condition = mask > 0.5  # adjust threshold if needed

# Create black background
black_bg = np.zeros(image.shape, dtype=np.uint8)

# Combine person and black background
output_image = np.where(condition[..., None], image, black_bg)

cv2.imwrite('normal2.png', output_image)
print("Image saved as output_black_bg.jpg")


