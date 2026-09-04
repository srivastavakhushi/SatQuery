import os
import numpy as np
import rasterio


BANDS = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B08", "B8A", "B09",
    "B10", "B11", "B12"
]


def get_band(file_path):
    """Read one Sentinel-2 band."""
    with rasterio.open(file_path) as src:
        return src.read(1)


def load_band_stack(image_folder):
    """
    Load all Sentinel-2 bands and stack them.

    Returns:
        NumPy array with shape:
        (13, height, width)
    """

    band_arrays = []

    for band_name in BANDS:

        file_path = os.path.join(
            image_folder,
            f"{band_name}.tif"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing band: {file_path}"
            )

        band = get_band(file_path)
        band_arrays.append(band)

    # Check that all bands have the same spatial dimensions
    first_shape = band_arrays[0].shape

    for band in band_arrays:
        if band.shape != first_shape:
            raise ValueError(
                f"Band dimensions do not match: "
                f"{band.shape} vs {first_shape}"
            )

    stack = np.stack(band_arrays, axis=0)

    return stack


if __name__ == "__main__":

    image_folder = (
        "data/Onera Satellite Change Detection dataset - Images/"
        "train_000/imgs_1_rect"
    )

    stack = load_band_stack(image_folder)

    print("Band stack shape:", stack.shape)
    print("Data type:", stack.dtype)
    print("Number of bands:", stack.shape[0])
    print("Image height:", stack.shape[1])
    print("Image width:", stack.shape[2])

    for i, band_name in enumerate(BANDS):
        print(
            f"{band_name}: "
            f"min={stack[i].min()}, "
            f"max={stack[i].max()}"
        )