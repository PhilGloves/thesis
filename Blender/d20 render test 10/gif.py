from PIL import Image
import os

input_folder = r"C:\Users\filip\Documents\thesis\Blender\renders_test10"
output_gif = r"C:\Users\filip\Documents\thesis\Blender\hologram.gif"

images = []

files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

for f in files:
    img = Image.open(os.path.join(input_folder, f))
    images.append(img)

images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=80,  # ms per frame
    loop=0
)

print("GIF creata:", output_gif)