
import os
from PIL import Image
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Path to the data directory
data_dir = Path(r"F:\Python\vision\captcha_images_v5/")
data_dir2 = Path(r"F:\Python\vision\captcha_images_v7/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
#labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

print("Number of images found: ", len(images))

for img in images:
    #print(img)
    im = Image.open(img).convert('RGBA')

    im.save(img.replace('captcha_images_v5','captcha_images_v7'))

# Load the image and convert to 32-bit RGBA
#im = Image.open(r"F:\Python\vision\captcha_images_v5\2bepe.png").convert('RGBA')

# Save result
#im.save(r"F:\Python\vision\captcha_images_v7\2bepe.png")


