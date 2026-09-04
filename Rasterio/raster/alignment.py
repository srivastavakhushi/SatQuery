import cv2
import numpy as np


def check_alignment(image1, image2):
    """
    Check whether two images have matching dimensions
    and calculate their pixel-wise difference.
    """

    if image1.shape != image2.shape:
        raise ValueError(
            f"Images have different shapes: "
            f"{image1.shape} vs {image2.shape}"
        )

    # Convert to float32 for numerical operations
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Calculate absolute pixel difference
    difference = cv2.absdiff(image1, image2)

    # Calculate average difference
    mean_difference = np.mean(difference)

    # Calculate maximum difference
    max_difference = np.max(difference)

    return mean_difference, max_difference


if __name__ == "__main__":

    from raster.temporal import load_temporal_pair

    image1, image2 = load_temporal_pair("train_000")

    # B04 is band index 3 because Python uses zero-based indexing
    b04_image1 = image1[3]
    b04_image2 = image2[3]

    mean_difference, max_difference = check_alignment(
        b04_image1,
        b04_image2
    )

    print("OpenCV alignment check")
    print("----------------------")
    print("Image shape:", b04_image1.shape)
    print("Mean pixel difference:", mean_difference)
    print("Maximum pixel difference:", max_difference)