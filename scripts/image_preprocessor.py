import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from config import train_image_dir, test_image_dir, image_size

def load_image(img_path, size=image_size):
    if not os.path.exists(img_path):
        return np.zeros((size[0], size[1], 3))
    img = Image.open(img_path).convert('RGB').resize(size)
    return np.asarray(img /255.0, dtype=np.uint8) 

def preprocess_images(df, image_dir, desc="Images", batch_size=32, size=(224,224)):
    """Yield batches of images instead of loading all into memory."""
    batch = []
    for sid in tqdm(df['sample_id'], desc=desc):
        img_path = os.path.join(image_dir, f"{sid}.jpg")
        batch.append(load_image(img_path, size=size))
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)