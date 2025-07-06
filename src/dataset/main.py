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
import pandas as pd

BATCH_SIZE = 4
OUT_DIR = "bird_dataset_by_species"

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
                return Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        return None
    

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        print("Fetching images...")
        return await tqdm_asyncio.tqdm.gather(*tasks)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    species_df = pd.read_csv("data/birds_poland.csv", sep="\t")
    polish_species = list(species_df["sci_name"])

    for species in polish_species:
        print("-------------------------------------------------------------------------")
        print("Species:", species)

        observations = get_observations(1, 3, species)
        if len(observations) > 0:
            imgs_data: List[ImageData] = []
            for obs in observations:
                imgs_data.extend(get_meta_from_observation(obs, species))

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
        
        print()