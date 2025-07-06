from tqdm import tqdm
import shutil

from pathlib import Path

input_dir = Path("bird_dataset_by_species")
output_dir = Path("bird_image_folder")

output_dir.mkdir(parents=True, exist_ok=True)

for file in tqdm(list(input_dir.iterdir())):
    if file.suffix.lower() != ".jpg":
        continue
    
    class_name = "_".join(file.stem.split("_")[:2])
    class_dir = output_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(file, class_dir / file.name)