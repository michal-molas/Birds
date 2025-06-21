from typing import List, Tuple
from pydantic import BaseModel, ConfigDict

from PIL.Image import Image

class ImageData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    img: Image
    species: str
    user: str
    license: str
    source_url: str 
    filename: str
    img_url: str