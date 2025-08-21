import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk

# ----------------- Utility Functions -----------------

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)

def srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0, 1)
    return np.where(srgb <= 0.04045,
                    srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308,
                    linear * 12.92,
                    1.055 * (linear ** (1 / 2.4)) - 0.055)

def relight(albedo, normals, light_x, light_y, light_z, light_color):
    normals = normalize(normals)
    light_dir = normalize(np.array([light_x, light_y, light_z]))
    view_dir = np.array([0, 0, 1.0])

    ambient_strength = 0.1
    diffuse_strength = 1.0
    specular_strength = 0.4
    shininess = 1

    ambient = ambient_strength * albedo * light_color

    dot_nl = np.clip(np.sum(normals * light_dir, axis=-1, keepdims=True), 0, 1)
    diffuse = diffuse_strength * dot_nl * albedo * light_color

    reflect_dir = -(2 * dot_nl * normals - light_dir)
    reflect_dir = normalize(reflect_dir)
    dot_rv = np.clip(np.sum(reflect_dir * view_dir, axis=-1, keepdims=True), 0, 1)
    specular = specular_strength * (dot_rv ** shininess) * light_color

    final = ambient + diffuse + specular
    return np.clip(final, 0, 1)  # Linear RGB in [0, 1]

# ----------------- GUI Functions -----------------

def update_image():
    lx = light_x.get() / 100.0
    ly = light_y.get() / 100.0
    lz = light_z.get() / 100.0
    lc_r = light_r.get() / 100.0
    lc_g = light_g.get() / 100.0
    lc_b = light_b.get() / 100.0
    light_color = np.array([lc_r, lc_g, lc_b])

    result_linear = relight(albedo_img.copy(), normal_img.copy(), lx, ly, lz, light_color)
    result_srgb = linear_to_srgb(result_linear)

    # Blend with original image using the mask
    blended = original_img * (1 - mask_img) + result_srgb * mask_img
    blended = np.clip(blended, 0, 1)

    img = Image.fromarray((blended * 255).astype(np.uint8))
    imgtk = ImageTk.PhotoImage(image=img)
    panel.config(image=imgtk)
    panel.image = imgtk

def set_values_from_entries():
    try:
        lx = float(entry_lx.get())
        ly = float(entry_ly.get())
        lz = float(entry_lz.get())
        lr = float(entry_lr.get())
        lg = float(entry_lg.get())
        lb = float(entry_lb.get())

        light_x.set(int(lx * 100))
        light_y.set(int(ly * 100))
        light_z.set(int(lz * 100))
        light_r.set(int(lr * 100))
        light_g.set(int(lg * 100))
        light_b.set(int(lb * 100))

        update_image()
    except ValueError:
        print("Invalid input in manual fields.")

# ----------------- Load Images -----------------

# Albedo
albedo_srgb = cv2.imread("albedo_image.png")
if albedo_srgb is None:
    raise FileNotFoundError("albedo_image.png not found")
albedo_srgb = cv2.cvtColor(albedo_srgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
albedo_img = srgb_to_linear(albedo_srgb)

# Normals
normal_img = cv2.imread("normal_image.png")
if normal_img is None:
    raise FileNotFoundError("normal_image.png not found")
normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
normal_img = cv2.resize(normal_img, (albedo_img.shape[1], albedo_img.shape[0]))
normal_img = normal_img.astype(np.float32) / 255.0 * 2.0 - 1.0

# Original image
original_img = cv2.imread("pic2.png")
if original_img is None:
    raise FileNotFoundError("pic2.png not found")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
original_img = cv2.resize(original_img, (albedo_img.shape[1], albedo_img.shape[0]))

# Mask
mask_img = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if mask_img is None:
    raise FileNotFoundError("mask.png not found")
mask_img = cv2.resize(mask_img, (albedo_img.shape[1], albedo_img.shape[0]))
mask_img = (mask_img > 128).astype(np.float32)[..., np.newaxis]  # Shape: (H, W, 1)

# ----------------- Tkinter GUI -----------------

root = tk.Tk()
root.title("Phong Relighting with Face Mask")

paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
paned.pack(fill=tk.BOTH, expand=True)

image_frame = tk.Frame(paned)
panel = tk.Label(image_frame)
panel.pack()
paned.add(image_frame)

control_frame = tk.Frame(paned)
paned.add(control_frame)

# Light direction sliders
light_x = Scale(control_frame, from_=-100, to=100, orient=HORIZONTAL, label="Light X", command=lambda _: update_image())
light_x.set(50)
light_x.pack(fill="x")

light_y = Scale(control_frame, from_=-100, to=100, orient=HORIZONTAL, label="Light Y", command=lambda _: update_image())
light_y.set(50)
light_y.pack(fill="x")

light_z = Scale(control_frame, from_=0, to=200, orient=HORIZONTAL, label="Light Z", command=lambda _: update_image())
light_z.set(100)
light_z.pack(fill="x")

# Light color sliders
light_r = Scale(control_frame, from_=0, to=100, orient=HORIZONTAL, label="Light R", command=lambda _: update_image())
light_r.set(100)
light_r.pack(fill="x")

light_g = Scale(control_frame, from_=0, to=100, orient=HORIZONTAL, label="Light G", command=lambda _: update_image())
light_g.set(100)
light_g.pack(fill="x")

light_b = Scale(control_frame, from_=0, to=100, orient=HORIZONTAL, label="Light B", command=lambda _: update_image())
light_b.set(100)
light_b.pack(fill="x")

# Manual input fields
entry_frame = tk.Frame(control_frame)
entry_frame.pack(pady=10)

tk.Label(entry_frame, text="Light X").grid(row=0, column=0)
entry_lx = tk.Entry(entry_frame, width=5)
entry_lx.insert(0, "0.5")
entry_lx.grid(row=0, column=1)

tk.Label(entry_frame, text="Light Y").grid(row=0, column=2)
entry_ly = tk.Entry(entry_frame, width=5)
entry_ly.insert(0, "0.5")
entry_ly.grid(row=0, column=3)

tk.Label(entry_frame, text="Light Z").grid(row=0, column=4)
entry_lz = tk.Entry(entry_frame, width=5)
entry_lz.insert(0, "1.0")
entry_lz.grid(row=0, column=5)

tk.Label(entry_frame, text="Light R").grid(row=1, column=0)
entry_lr = tk.Entry(entry_frame, width=5)
entry_lr.insert(0, "1.0")
entry_lr.grid(row=1, column=1)

tk.Label(entry_frame, text="Light G").grid(row=1, column=2)
entry_lg = tk.Entry(entry_frame, width=5)
entry_lg.insert(0, "1.0")
entry_lg.grid(row=1, column=3)

tk.Label(entry_frame, text="Light B").grid(row=1, column=4)
entry_lb = tk.Entry(entry_frame, width=5)
entry_lb.insert(0, "1.0")
entry_lb.grid(row=1, column=5)

apply_button = tk.Button(control_frame, text="Apply Manual Values", command=set_values_from_entries)
apply_button.pack(pady=5)

# Initial display
initial_light_color = np.array([1.0, 1.0, 1.0])
initial_result = relight(albedo_img.copy(), normal_img.copy(), 0.5, 0.5, 1.0, initial_light_color)
initial_result = linear_to_srgb(initial_result)
blended = original_img * (1 - mask_img) + initial_result * mask_img
blended = np.clip(blended, 0, 1)

img = Image.fromarray((blended * 255).astype(np.uint8))
imgtk = ImageTk.PhotoImage(image=img)
panel.config(image=imgtk)
panel.image = imgtk

root.mainloop()
