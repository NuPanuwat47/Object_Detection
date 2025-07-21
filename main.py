import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import numpy as np
import pathlib

# สำหรับระบบ Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# โหลดโมเดล YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

def resize_and_pad(image, target_size=(640, 480), bg_color=(240, 244, 247)):
    """Resize image to fit in target_size, keep aspect ratio, pad with bg_color."""
    img_w, img_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img
# --- ฟังก์ชันเลือกภาพ ---
def select_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file_path:
        return

    image = Image.open(file_path)
    image = resize_and_pad(image)
    img_tk = ImageTk.PhotoImage(image)

    result_label.config(image=img_tk)
    result_label.image = img_tk
    result_label.file_path = file_path

# --- ฟังก์ชันทำนายภาพ ---
def predict_image():
    if not hasattr(result_label, 'file_path') or not result_label.file_path:
        messagebox.showwarning("No Image", "Please select an image first.")
        return

    file_path = result_label.file_path
    image = Image.open(file_path)
    img_array = np.array(image)

    results = model(img_array)
    img_result = results.render()[0]
    img_result_pil = Image.fromarray(img_result)
    img_result_pil = resize_and_pad(img_result_pil)

    img_result_tk = ImageTk.PhotoImage(img_result_pil)

    result_label.config(image=img_result_tk)
    result_label.image = img_result_tk

    num_detected = len(results.xyxy[0])
    if num_detected > 0:
        messagebox.showinfo("Detected Result", f"ตรวจพบวัตถุ {num_detected} รายการ")
    else:
        messagebox.showinfo("Detected Result", "ไม่พบวัตถุในภาพนี้")


root = tk.Tk()
root.title("🧠 Object Detection with YOLOv5")
root.geometry("850x750")
root.configure(bg="#f0f4f7") 

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12), padding=10)
style.configure("TLabel", font=("Segoe UI", 12), background="#f0f4f7")

title_label = ttk.Label(root, text="Object Detection", font=("Segoe UI", 18, "bold"))
title_label.pack(pady=20)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

select_button = ttk.Button(button_frame, text="📁 Select Image", command=select_image)
select_button.grid(row=0, column=0, padx=10)

predict_button = ttk.Button(button_frame, text="🔍 Detect Objects", command=predict_image)
predict_button.grid(row=0, column=1, padx=10)

result_label = ttk.Label(root)
result_label.pack(pady=20)

footer = ttk.Label(root, text="Developed with YOLOv5 + Tkinter", font=("Segoe UI", 10, "italic"), foreground="#666")
footer.pack(side="bottom", pady=10)

root.mainloop()
