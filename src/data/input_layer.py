"""
Phase 1: Input/Compatibility layer
Validates and preprocesses satellite imagery (GeoTIFF/TIFF) before VLM inference.
"""
import rasterio
import numpy as np
from PIL import Image
import os


class SatelliteImageValidator:
    """Validates and normalizes satellite image inputs for VLM pipeline."""

    SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
    MAX_DIMENSION = 4096  # safety cap before resize

    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}

    def validate(self):
        """Run all validation checks. Returns (is_valid, report dict)."""
        report = {"filepath": self.filepath, "checks": {}}

        # 1. File exists
        if not os.path.exists(self.filepath):
            report["checks"]["file_exists"] = False
            return False, report
        report["checks"]["file_exists"] = True

        # 2. Extension check
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            report["checks"]["supported_format"] = False
            return False, report
        report["checks"]["supported_format"] = True
        report["extension"] = ext

        # 3. GeoTIFF-specific checks via rasterio
        if ext in {'.tif', '.tiff'}:
            try:
                with rasterio.open(self.filepath) as src:
                    report["band_count"] = src.count
                    report["width"] = src.width
                    report["height"] = src.height
                    report["crs"] = str(src.crs) if src.crs else None
                    report["dtype"] = str(src.dtypes[0])
                    report["checks"]["readable"] = True

                    # Modality check: single-band, RGB (3-band), or multispectral (>3 band)
                    if src.count == 1:
                        report["modality"] = "single_band"
                    elif src.count == 3:
                        report["modality"] = "rgb"
                    else:
                        report["modality"] = "multispectral"

                    # Resolution sanity check
                    if src.width < 32 or src.height < 32:
                        report["checks"]["min_resolution"] = False
                        return False, report
                    report["checks"]["min_resolution"] = True

            except Exception as e:
                report["checks"]["readable"] = False
                report["error"] = str(e)
                return False, report
        else:
            # Plain image (jpg/png) — no georeferencing expected
            try:
                with Image.open(self.filepath) as img:
                    report["width"], report["height"] = img.size
                    report["modality"] = "rgb"
                    report["crs"] = None
                    report["checks"]["readable"] = True
            except Exception as e:
                report["checks"]["readable"] = False
                report["error"] = str(e)
                return False, report

        self.metadata = report
        return True, report

    def to_vlm_ready_image(self, output_path="vlm_input.jpg"):
        """
        Converts validated input into a VLM-compatible RGB JPEG,
        resizing if needed to stay under safe pixel limits.
        """
        ext = os.path.splitext(self.filepath)[1].lower()

        if ext in {'.tif', '.tiff'}:
            with rasterio.open(self.filepath) as src:
                if src.count >= 3:
                    # take first 3 bands as RGB (band order assumption — refine per dataset later)
                    arr = src.read([1, 2, 3])
                else:
                    # single band — replicate to pseudo-RGB
                    band = src.read(1)
                    arr = np.stack([band, band, band])

                # normalize to 0-255 uint8
                arr = arr.astype(np.float32)
                for i in range(arr.shape[0]):
                    band = arr[i]
                    lo, hi = np.percentile(band, (2, 98))
                    band = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
                    arr[i] = band * 255
                arr = arr.astype(np.uint8)
                arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
                img = Image.fromarray(arr)
        else:
            img = Image.open(self.filepath).convert("RGB")

        img.thumbnail((self.MAX_DIMENSION, self.MAX_DIMENSION))
        img.save(output_path)
        return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python input_layer.py <path_to_image>")
        sys.exit(1)

    validator = SatelliteImageValidator(sys.argv[1])
    is_valid, report = validator.validate()
    print("Valid:", is_valid)
    print("Report:", report)

    if is_valid:
        out = validator.to_vlm_ready_image()
        print("VLM-ready image saved to:", out)
