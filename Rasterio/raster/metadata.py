import rasterio


def get_raster_metadata(file_path):
    with rasterio.open(file_path) as src:
        metadata = {
            "filename": file_path,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs),
            "resolution": src.res,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }

    return metadata


if __name__ == "__main__":
    file_path = "data/test.tif"

    metadata = get_raster_metadata(file_path)

    for key, value in metadata.items():
        print(f"{key}: {value}")