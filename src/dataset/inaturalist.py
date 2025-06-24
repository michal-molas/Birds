import requests
import os
import time
from PIL import Image
from tqdm import tqdm
from typing import List
import numpy as np

from .model import ImageData

OUT_DIR = "bird_dataset2"
os.makedirs(OUT_DIR, exist_ok=True)

def get_observations(start_page: int, pages: int, species: str) -> list:
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": 3,  # Birds
        # "place_id": 7800,  # Poland
        "place_id": 97391, # Europe
        "photos": "true",
        "per_page": 200,
        "order_by": "date_added",
        "order": "desc",
        "license": "CC0,CC-BY,CC-BY-SA,CC-BY-NC,CC-BY-NC-SA",
        "taxon_name": species,
    }
    all_obs = []

    print("Fetching observations from iNaturalist...")
    for page in tqdm(list(range(start_page, start_page + pages))):
        params["page"] = page
        res = requests.get(url, params=params).json()
        if len(res["results"]) == 0:
            break
        all_obs.extend(res["results"])
        time.sleep(1)

    return all_obs


def get_meta_from_observation(obs: dict, species: str) -> List[ImageData]:
    user = obs.get("user", {}).get("login", "unknown")
    license = obs.get("license_code", "unknown")
    obs_id = obs["id"]

    result = []

    photos = obs.get("photos", [])
    if len(photos) == 0:
        return []
    
    # I'm taking only first photo from each observation,
    # because photos in the same observation are similar
    # and may cause problems when splitting into training and validation sets
    photo = photos[0]

    url = photo["url"].replace("square", "original")
    filename = f"{species.replace(' ', '_')}_{obs_id}.jpg"

    result.append(ImageData(
        img=Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)),
        species=species,
        user=user,
        license=license,
        source_url=f"https://www.inaturalist.org/observations/{obs_id}",
        filename=filename,
        img_url=url,
    ))

    return result
