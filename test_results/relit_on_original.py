import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk

# ----------------- Utility Functions -----------------

def apply_hdr_relighting(albedo, diffuse, specular):
    albedo = albedo / 255.0  # Normalize albedo
    result = albedo * diffuse + specular
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

def update_image():
    # Get light color from sliders
    lc_r = light_r.get() / 100.0
    lc_g = light_g.get() / 100.0
    lc_b = light_b.get() / 100.0
    light_color = np.array([lc_r, lc_g, lc_b])

    # Apply colored lighting to diffuse and specular
    mod_diffuse = diffuse_img * light_color
    mod_specular = specular_img * light_color

    relit_face = apply_hdr_relighting(albedo_img.copy(), mod_diffuse, mod_specular)

    # Blend with original image using mask
    blended = original_img.copy()
    blended = blended * (1 - mask_img) + relit_face * mask_img
    blended = blended.astype(np.uint8)

    img = Image.fromarray(blended)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.config(image=imgtk)
    panel.image = imgtk

def set_values_from_entries():
    try:
        lr = float(entry_lr.get())
        lg = float(entry_lg.get())
        lb = float(entry_lb.get())

        light_r.set(int(lr * 100))
        light_g.set(int(lg * 100))
        light_b.set(int(lb * 100))

        update_image()
    except ValueError:
        print("Invalid input in manual fields.")

# ----------------- Load Images -----------------

original_img = cv2.imread("pic2.png")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

albedo_img = cv2.imread("albedo_image.png")
albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_BGR2RGB)

diffuse_img = cv2.imread("intrD.hdr", -1)
specular_img = cv2.imread("intrS.hdr", -1)

# Resize HDR maps to match albedo if needed
diffuse_img = cv2.resize(diffuse_img, (albedo_img.shape[1], albedo_img.shape[0]))
specular_img = cv2.resize(specular_img, (albedo_img.shape[1], albedo_img.shape[0]))

mask_img = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
mask_img = cv2.resize(mask_img, (albedo_img.shape[1], albedo_img.shape[0]))
mask_img = (mask_img > 128).astype(np.float32)[..., np.newaxis]

# ----------------- Tkinter GUI -----------------

root = tk.Tk()
root.title("HDR-Based Face Relighting")

paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
paned.pack(fill=tk.BOTH, expand=True)

image_frame = tk.Frame(paned)
panel = tk.Label(image_frame)
panel.pack()
paned.add(image_frame)

control_frame = tk.Frame(paned)
paned.add(control_frame)

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

entry_frame = tk.Frame(control_frame)
entry_frame.pack(pady=10)

# Manual light color entry
tk.Label(entry_frame, text="Light R").grid(row=0, column=0)
entry_lr = tk.Entry(entry_frame, width=5)
entry_lr.insert(0, "1.0")
entry_lr.grid(row=0, column=1)

tk.Label(entry_frame, text="Light G").grid(row=0, column=2)
entry_lg = tk.Entry(entry_frame, width=5)
entry_lg.insert(0, "1.0")
entry_lg.grid(row=0, column=3)

tk.Label(entry_frame, text="Light B").grid(row=0, column=4)
entry_lb = tk.Entry(entry_frame, width=5)
entry_lb.insert(0, "1.0")
entry_lb.grid(row=0, column=5)

apply_button = tk.Button(control_frame, text="Apply Manual Values", command=set_values_from_entries)
apply_button.pack(pady=5)

# Initial render
relit_initial = apply_hdr_relighting(albedo_img.copy(), diffuse_img, specular_img)
blended = original_img.copy()
blended = blended * (1 - mask_img) + relit_initial * mask_img
blended = blended.astype(np.uint8)

img = Image.fromarray(blended)
imgtk = ImageTk.PhotoImage(image=img)
panel.config(image=imgtk)
panel.image = imgtk

root.mainloop()
