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

def get_observations(start_page=1, pages=10):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": 3,  # Birds
        "place_id": 7800,  # Poland
        # "place_id": 97391, # Europe
        "photos": "true",
        "per_page": 100,
        "order_by": "date_added",
        "order": "asc",
        "license": "CC0,CC-BY,CC-BY-SA,CC-BY-NC,CC-BY-NC-SA",
    }
    all_obs = []

    print("Fetching observations from iNaturalist...")
    for page in tqdm(list(range(start_page, start_page + pages))):
        params["page"] = page
        res = requests.get(url, params=params).json()
        all_obs.extend(res["results"])
        time.sleep(1)

    return all_obs


def get_meta_from_observation(obs: dict) -> List[ImageData]:
    taxon = obs.get("taxon", {})
    sci_name = taxon.get("name", "unknown").replace(" ", "_")
    user = obs.get("user", {}).get("login", "unknown")
    license = obs.get("license_code", "unknown")
    obs_id = obs["id"]

    result = []

    for i, photo in enumerate(obs.get("photos", [])):
        url = photo["url"].replace("square", "original")
        filename = f"{sci_name}_{obs_id}_{i}.jpg"

        result.append(ImageData(
            img=Image.fromarray(np.zeros((5, 5))),
            species=sci_name,
            user=user,
            license=license,
            source_url=f"https://www.inaturalist.org/observations/{obs_id}",
            filename=filename,
            img_url=url,
        ))

    return result
