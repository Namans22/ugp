import numpy as np
import cv2
from PIL import Image

# Load images (Replace with your file paths)
mask_image = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Mask image (grayscale)
masked_image = cv2.imread('masked_face.png')  # Masked face image (BGR)
normal_map = cv2.imread('normal_image.png')  # Normal map (RGB)

# Normalize the normal map values to [-1, 1] range
normal_map = normal_map.astype(np.float32) / 255.0  # Normalize to [0, 1]
normals = 2.0 * normal_map - 1.0  # Convert to [-1, 1] range

# Define light direction (example: light coming from the right)
light_direction = np.array([0.8, 0, 0], dtype=np.float32)  # Light direction (X-axis)
light_direction = light_direction / np.linalg.norm(light_direction)  # Normalize

# Define light color (example: white light)
light_color = np.array([1, 0, 0], dtype=np.float32)  # RGB white light

# Define material properties
ka = 0.1  # Ambient coefficient
kd = 0.7  # Diffuse coefficient
ks = 0.8  # Specular coefficient
shininess = 10  # Shininess exponent

# Calculate the ambient light
ambient = ka * light_color

# Function to compute diffuse and specular reflections using Phong shading for a single pixel
def compute_phong_lighting(normal, light_direction, light_color, diffuse_texture, specular_texture):
    # Diffuse reflection: Lambertian shading (cosine of the angle between normal and light)
    diffuse_intensity = max(np.dot(normal, light_direction), 0)
    
    # Specular reflection: Phong reflection model check this 
    reflect_direction = 2 * diffuse_intensity * normal - light_direction
    view_direction = np.array([0, 0, 1], dtype=np.float32)  # Assuming the camera is at origin
    specular_intensity = max(np.dot(reflect_direction, view_direction), 0) ** shininess

    # Calculate combined lighting (ambient, diffuse, and specular)
    ambient_light = ka * light_color
    diffuse_light = kd * diffuse_intensity * diffuse_texture  # Scalar * vector
    specular_light = ks * specular_intensity * specular_texture  # Scalar * vector
    
    final_color = ambient_light + diffuse_light + specular_light
    return np.clip(final_color, 0, 1)

# Load diffuse and specular textures (using albedo and specular maps)
diffuse_texture = cv2.imread('albedo_image.png').astype(np.float32) / 255.0  # Normalize (RGB image)
specular_texture = cv2.imread('intrS.hdr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # HDR specular texture

# If the specular texture is HDR, normalize it (note: HDR textures may have a range [0, inf] instead of [0, 1])
specular_texture = specular_texture / np.max(specular_texture)  # Normalize HDR specular texture

# If the diffuse texture is grayscale, convert it to RGB by duplicating the values
if len(diffuse_texture.shape) == 2:  # It's a grayscale image
    diffuse_texture = np.repeat(diffuse_texture[:, :, np.newaxis], 3, axis=2)  # Convert to RGB by repeating

# Create an empty result image (same size as the masked face)
result_image = np.zeros_like(masked_image, dtype=np.float32)

# Apply Phong shading model to the face region (based on the mask)
for i in range(masked_image.shape[0]):
    for j in range(masked_image.shape[1]):
        if mask_image[i, j] > 0:  # If the pixel is part of the face region (non-zero mask)
            normal = normals[i, j]
            # Get the corresponding pixel from the diffuse and specular textures
            diffuse_pixel = diffuse_texture[i, j, :]  # RGB vector
            specular_pixel = specular_texture[i, j, :]  # RGB vector
            
            # Calculate the relit color using Phong shading model
            color = compute_phong_lighting(normal, light_direction, light_color, diffuse_pixel, specular_pixel)
            result_image[i, j] = color

# Convert the result image to a displayable format (clamp to [0, 255] and convert to uint8)
result_image = (result_image * 255).astype(np.uint8)

# Convert to BGR for OpenCV (as OpenCV uses BGR format)
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR



# Optionally, save the relit image to disk
cv2.imwrite('relit_face.png', result_image_bgr)