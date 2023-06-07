from pydantic import BaseModel


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    model_dir = "model"
