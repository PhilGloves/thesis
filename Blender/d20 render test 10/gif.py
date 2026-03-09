from pathlib import Path
from PIL import Image

input_folder = Path.cwd()
output_gif = input_folder / f"{input_folder.name}.gif"

files = sorted(input_folder.glob("*.png"))

if not files:
    raise FileNotFoundError(f"Nessun PNG trovato in: {input_folder}")

forward_images = [Image.open(file).copy() for file in files]
backward_images = forward_images[-2:0:-1]
all_images = forward_images + backward_images

all_images[0].save(
    output_gif,
    save_all=True,
    append_images=all_images[1:],
    duration=80,
    loop=0
)

print("GIF creata:", output_gif)