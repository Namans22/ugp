import numpy as np
import cv2
import matplotlib.pyplot as plt

def srgb_to_linear(srgb):
    """Convert sRGB values (in [0, 1]) to linear RGB."""
    srgb = np.clip(srgb, 0, 1)
    linear = np.where(srgb <= 0.04045,
                      srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    return linear

# Load and convert to linear RGB
albedo_srgb = cv2.imread('albedo_image.png').astype(np.float32) / 255.0
albedo_srgb = cv2.cvtColor(albedo_srgb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
albedo_linear = srgb_to_linear(albedo_srgb)

# Display the linear image using matplotlib
plt.imshow(np.clip(albedo_linear, 0, 1))
plt.title("Linear RGB Albedo Image")
plt.axis('off')
plt.show()
