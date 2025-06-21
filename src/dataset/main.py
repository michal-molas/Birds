from tqdm import tqdm
import tqdm.asyncio as tqdm_asyncio 
from typing import List
from .model import ImageData
import os
import json
from .inaturalist import (
    get_observations,
    get_meta_from_observation,
)
from .bird_extraction import detect_bird_frcnn
import aiohttp
import asyncio
from io import BytesIO
from PIL import Image

BATCH_SIZE = 8
OUT_DIR = "bird_dataset"

def save_img(img_data: ImageData | None):
    if img_data is None:
        return
    
    path = os.path.join(OUT_DIR, img_data.filename)
    img_data.img.save(path)

    meta = {
        "image": img_data.filename,
        "species": img_data.species,
        "user": img_data.user,
        "license": img_data.license,
        "source_url": img_data.source_url
    }
    with open(path.replace(".jpg", ".json"), "w") as f:
        json.dump(meta, f)
    

async def fetch(session, url):
    try:
        async with session.get(url, timeout=200) as resp:
            if resp.status == 200:
                content = await resp.read()
                return Image.open(BytesIO(content))
    except Exception as e:
        return None
    

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        print("Fetching images...")
        return await tqdm_asyncio.tqdm.gather(*tasks)



if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    observations = get_observations(1, 50)
    imgs_data: List[ImageData] = []
    for obs in observations:
        imgs_data.extend(get_meta_from_observation(obs))

    all_urls = [img_data.img_url for img_data in imgs_data]
    all_imgs = asyncio.run(fetch_all(all_urls))

    for i, img_data in enumerate(imgs_data):
        if all_imgs[i] is None:
            continue
        
        img_data.img = all_imgs[i]

    print("Processing and saving images...")
    for i in tqdm(list(range(0, len(imgs_data), BATCH_SIZE))):
        cropped_batch = detect_bird_frcnn(imgs_data[i:i + BATCH_SIZE])
        for c_img in cropped_batch:
            save_img(c_img)