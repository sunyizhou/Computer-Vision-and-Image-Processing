import numpy as np
from PIL import Image
import os

def load_images(path):
    images = []
    for _file in os.listdir(path):
        if _file.endswith('.jpg'):
            #print(os.path.join(path, _file))
            img = np.array(Image.open((os.path.join(path, _file))).convert("L"), dtype=np.float32)
            img /= img.max()
            images.append(img)
    return images