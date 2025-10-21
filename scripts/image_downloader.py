import os
from tqdm import tqdm
from config import train_image_dir, test_image_dir,test_csv,train_csv

import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib

def download_image(image_link, savefolder):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

def download_images_with_retry(image_links, download_folder, max_retries=3):
    os.makedirs(download_folder, exist_ok=True)
    for _ in range(max_retries):
        try:
            download_images(image_links, download_folder)
            break
        except Exception as e:
            print(f"Image download failed for {download_folder}, retrying... Error: {e}")
    print(f"Images downloaded to {download_folder}")

def download_all_images(train_df, test_df):
    download_images_with_retry(train_df['image_link'].tolist(), train_image_dir)
    # download_images_with_retry(test_df['image_link'].tolist(), test_image_dir)

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
download_all_images(train_df, test_df)