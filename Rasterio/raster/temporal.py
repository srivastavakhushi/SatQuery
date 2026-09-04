import os

from raster.bands import load_band_stack


DATASET_PATH = "data/Onera Satellite Change Detection dataset - Images"


def load_temporal_pair(sample_name):
    """
    Load the two temporal images for one OSCD100 sample.

    Returns:
        image1: NumPy array of shape (13, height, width)
        image2: NumPy array of shape (13, height, width)
    """

    sample_path = os.path.join(DATASET_PATH, sample_name)

    image1_path = os.path.join(sample_path, "imgs_1_rect")
    image2_path = os.path.join(sample_path, "imgs_2_rect")

    if not os.path.exists(image1_path):
        raise FileNotFoundError(
            f"Image 1 folder not found: {image1_path}"
        )

    if not os.path.exists(image2_path):
        raise FileNotFoundError(
            f"Image 2 folder not found: {image2_path}"
        )

    image1 = load_band_stack(image1_path)
    image2 = load_band_stack(image2_path)

    if image1.shape != image2.shape:
        raise ValueError(
            f"Temporal images have different shapes: "
            f"{image1.shape} vs {image2.shape}"
        )

    return image1, image2


if __name__ == "__main__":

    image1, image2 = load_temporal_pair("train_000")

    print("Temporal pair loaded successfully")
    print()
    print("Image 1 shape:", image1.shape)
    print("Image 2 shape:", image2.shape)
    print("Image 1 dtype:", image1.dtype)
    print("Image 2 dtype:", image2.dtype)

    print()
    print("Number of bands:", image1.shape[0])
    print("Height:", image1.shape[1])
    print("Width:", image1.shape[2])

    print()
    print("Temporal pair validation: PASSED")