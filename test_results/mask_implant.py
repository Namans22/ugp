import cv2
import numpy as np

# Load original image (the full one)
original_img = cv2.imread("pic2.png")  # Replace with your actual file
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Load relit face result (from your relight output)
relit_face = cv2.imread("relit_result.png")  # Save your GUI output or modify code to extract it directly
relit_face = cv2.cvtColor(relit_face, cv2.COLOR_BGR2RGB)

# Load the face mask
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

# Threshold the mask to binary
_, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

# Resize relit face to match original image size if needed
relit_face = cv2.resize(relit_face, (original_img.shape[1], original_img.shape[0]))

# Create inverse mask
inv_mask = cv2.bitwise_not(binary_mask)

# Mask out face region in original image
background = cv2.bitwise_and(original_img, original_img, mask=inv_mask)

# Mask out non-face region in relit image
foreground = cv2.bitwise_and(relit_face, relit_face, mask=binary_mask)

# Combine both
blended = cv2.add(background, foreground)

# Save or display
cv2.imwrite("relit_face_on_original.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
