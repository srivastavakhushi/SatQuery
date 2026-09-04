import numpy as np


def validate_temporal_pair(image1, image2):
    """
    Validate that two temporal raster arrays have compatible dimensions.
    """

    if image1.shape != image2.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"{image1.shape} vs {image2.shape}"
        )

    return True


def normalize_band(band, method="minmax"):
    """
    Normalize a raster band using the selected method.

    Parameters
    ----------
    band : numpy.ndarray
        Input raster band.

    method : str
        Normalization method:
        - "minmax": scale values to [0, 1]
        - "none": keep original values, converted to float32

    Returns
    -------
    numpy.ndarray
        Preprocessed float32 array.
    """

    # Convert input to float32
    band = band.astype(np.float32)

    # No normalization
    if method == "none":
        return band

    # Min-max normalization
    if method == "minmax":

        min_value = band.min()
        max_value = band.max()

        # Prevent division by zero
        if max_value == min_value:
            return np.zeros_like(
                band,
                dtype=np.float32
            )

        normalized = (
            (band - min_value)
            / (max_value - min_value)
        )

        return normalized

    # Invalid method
    raise ValueError(
        f"Unknown normalization method: {method}. "
        f"Use 'minmax' or 'none'."
    )


if __name__ == "__main__":

    import rasterio

    base = (
        "data/Onera Satellite Change Detection dataset - Images/"
        "train_000"
    )

    image1_path = f"{base}/imgs_1_rect/B04.tif"
    image2_path = f"{base}/imgs_2_rect/B04.tif"

    # -------------------------------------------------
    # 1. Read B04 from both temporal images
    # -------------------------------------------------

    with rasterio.open(image1_path) as src:
        image1 = src.read(1)

    with rasterio.open(image2_path) as src:
        image2 = src.read(1)

    print("Image 1 shape:", image1.shape)
    print("Image 2 shape:", image2.shape)

    # -------------------------------------------------
    # 2. Validate temporal pair
    # -------------------------------------------------

    validate_temporal_pair(image1, image2)

    # -------------------------------------------------
    # 3. Select preprocessing method
    # -------------------------------------------------

    normalization_method = "minmax"

    # -------------------------------------------------
    # 4. Preprocess both images
    # -------------------------------------------------

    image1_normalized = normalize_band(
        image1,
        method=normalization_method
    )

    image2_normalized = normalize_band(
        image2,
        method=normalization_method
    )

    # -------------------------------------------------
    # 5. Display results
    # -------------------------------------------------

    print("\nImage 1:")
    print("  Original dtype:", image1.dtype)
    print("  Original min:", image1.min())
    print("  Original max:", image1.max())

    print("  Normalized dtype:", image1_normalized.dtype)
    print("  Normalized min:", image1_normalized.min())
    print("  Normalized max:", image1_normalized.max())

    print("\nImage 2:")
    print("  Original dtype:", image2.dtype)
    print("  Original min:", image2.min())
    print("  Original max:", image2.max())

    print("  Normalized dtype:", image2_normalized.dtype)
    print("  Normalized min:", image2_normalized.min())
    print("  Normalized max:", image2_normalized.max())

    print("\nPreprocessing method:", normalization_method)
    print("Temporal pair validation: PASSED")
    print("Preprocessing test: PASSED")