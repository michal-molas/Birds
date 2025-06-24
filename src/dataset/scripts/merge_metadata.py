import os
import json
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    with open("metadata.jsonl", "w", encoding="utf-8") as out_file:
        for path in tqdm(glob(os.path.join("bird_dataset_by_species", "*.json"))):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                out_file.write(json.dumps(data, ensure_ascii=False) + "\n")