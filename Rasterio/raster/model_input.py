from dataclasses import dataclass

import numpy as np

from raster.temporal import load_temporal_pair
from raster.preprocessing import normalize_band


BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12"
]


@dataclass
class ModelInput:
    """
    Standardized input passed from the raster pipeline
    to an AI model.
    """

    sample_id: str
    image1: np.ndarray
    image2: np.ndarray
    band_names: list
    preprocessing_method: str

    @property
    def shape(self):
        return self.image1.shape


def preprocess_image(image):
    """
    Normalize every band independently.
    """

    processed_bands = []

    for band in image:
        normalized = normalize_band(band)
        processed_bands.append(normalized)

    return np.stack(processed_bands, axis=0)


def prepare_model_input(sample_name):
    """
    Load and preprocess both temporal images.
    """

    image1, image2 = load_temporal_pair(sample_name)

    if image1.shape != image2.shape:
        raise ValueError(
            f"Temporal images have different shapes: "
            f"{image1.shape} vs {image2.shape}"
        )

    image1_processed = preprocess_image(image1)
    image2_processed = preprocess_image(image2)

    return ModelInput(
        sample_id=sample_name,
        image1=image1_processed,
        image2=image2_processed,
        band_names=BAND_NAMES,
        preprocessing_method="per-band-minmax"
    )


if __name__ == "__main__":

    model_input = prepare_model_input("train_000")

    print("MODEL INPUT")
    print("-----------")

    print("Sample ID:", model_input.sample_id)

    print("Image 1 shape:", model_input.image1.shape)
    print("Image 2 shape:", model_input.image2.shape)

    print("Image 1 dtype:", model_input.image1.dtype)
    print("Image 2 dtype:", model_input.image2.dtype)

    print("Image 1 range:",
          model_input.image1.min(),
          "to",
          model_input.image1.max())

    print("Image 2 range:",
          model_input.image2.min(),
          "to",
          model_input.image2.max())

    print()
    print("Bands:", model_input.band_names)
    print("Number of bands:", len(model_input.band_names))
    print("Preprocessing:", model_input.preprocessing_method)

    print()
    print("Temporal dimensions match:",
          model_input.image1.shape == model_input.image2.shape)

    print("Model input preparation: PASSED")