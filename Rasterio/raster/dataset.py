import os

from raster.temporal import load_temporal_pair


DATASET_PATH = "data/Onera Satellite Change Detection dataset - Images"


def get_sample_names():
    """
    Find all OSCD100 samples containing the expected
    temporal image folders.
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    sample_names = []

    for name in sorted(os.listdir(DATASET_PATH)):

        sample_path = os.path.join(
            DATASET_PATH,
            name
        )

        image1_path = os.path.join(
            sample_path,
            "imgs_1_rect"
        )

        image2_path = os.path.join(
            sample_path,
            "imgs_2_rect"
        )

        if (
            os.path.isdir(sample_path)
            and os.path.isdir(image1_path)
            and os.path.isdir(image2_path)
        ):
            sample_names.append(name)

    return sample_names


def validate_dataset():
    """
    Load every OSCD100 sample and verify that
    both temporal images have matching dimensions.
    """

    sample_names = get_sample_names()

    if not sample_names:
        raise ValueError(
            "No valid OSCD100 samples were found."
        )

    successful = 0
    failed = []

    for sample_name in sample_names:

        try:
            image1, image2 = load_temporal_pair(
                sample_name
            )

            if image1.shape != image2.shape:
                raise ValueError(
                    f"Shape mismatch: "
                    f"{image1.shape} vs {image2.shape}"
                )

            if image1.shape[0] != 13:
                raise ValueError(
                    f"Expected 13 bands, "
                    f"found {image1.shape[0]}"
                )

            successful += 1

            print(
                f"[PASSED] {sample_name} "
                f"{image1.shape}"
            )

        except Exception as error:

            failed.append(
                (sample_name, str(error))
            )

            print(
                f"[FAILED] {sample_name}: "
                f"{error}"
            )

    print()
    print("DATASET VALIDATION")
    print("------------------")
    print("Total samples:", len(sample_names))
    print("Successful:", successful)
    print("Failed:", len(failed))

    if failed:
        print()
        print("Failures:")

        for sample_name, error in failed:
            print(
                f"  {sample_name}: {error}"
            )

    return {
        "total": len(sample_names),
        "successful": successful,
        "failed": failed
    }


if __name__ == "__main__":

    result = validate_dataset()

    print()

    if result["failed"]:
        print("Dataset validation: FAILED")
    else:
        print("Dataset validation: PASSED")