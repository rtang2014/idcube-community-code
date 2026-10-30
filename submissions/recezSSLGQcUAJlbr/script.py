"""Convert original ZY1E VNIR/SWIR TIFFs into wavelength-labelled ENVI pairs.

All 76 VNIR and 90 SWIR calibrated wavelengths are embedded below.
No separate wavelength files are needed. Original TIFFs remain unchanged.

Install: py -m pip install rasterio numpy
Run:     py convert_zy1e_embedded.py
Or:      py convert_zy1e_embedded.py --vn "path/to/VN.tiff" --sw "path/to/SW.tiff"
"""
import argparse
import csv
from pathlib import Path
import sys


# Calibrated center wavelengths (nm) from your uploaded spectral-response files.
WAVELENGTHS_NM = {
    "VNIR": [
        395.8612, 404.4524, 413.0436, 421.6348, 430.226,
        438.583801, 447.098741, 455.786167, 464.365594, 473.208972,
        481.829453, 490.449933, 499.070413, 507.690893, 516.311373,
        524.931853, 533.552334, 542.024472, 550.581904, 559.234311,
        567.886719, 576.477238, 585.067758, 593.658277, 602.248797,
        610.839317, 619.429836, 628.020356, 636.610876, 645.209401,
        653.787764, 662.286899, 670.786035, 679.387819, 687.989604,
        696.591389, 705.193173, 713.794958, 722.396743, 730.998527,
        739.578525, 748.158522, 756.73852, 765.440069, 774.001122,
        782.562176, 791.12323, 799.684284, 808.245338, 816.806392,
        825.367446, 833.873589, 842.475202, 851.076816, 859.67843,
        868.280043, 876.881657, 885.483271, 894.084884, 902.686498,
        911.288112, 919.889726, 928.491339, 937.092953, 945.711952,
        954.243026, 962.774099, 971.305173, 979.836246, 988.367319,
        996.898393, 1005.8364, 1014.4276, 1023.0188, 1031.61,
        1040.2012,
    ],
    "SWIR": [
        1005.60375, 1022.2125, 1039.19175, 1055.87195, 1072.55215,
        1089.23235, 1105.91255, 1122.59275, 1139.444, 1156.41275,
        1173.28985, 1190.16695, 1207.04405, 1223.92115, 1240.79825,
        1257.3965, 1274.148375, 1290.90025, 1307.652125, 1324.404,
        1341.155875, 1357.90775, 1374.867, 1391.65125, 1408.491977,
        1425.332705, 1442.173432, 1459.014159, 1475.854886, 1492.695614,
        1509.536341, 1526.377068, 1543.217795, 1560.058523, 1576.89925,
        1593.714, 1610.39, 1627.1325, 1644.011281, 1660.890063,
        1677.768844, 1694.647625, 1711.526406, 1728.405188, 1745.283969,
        1762.16275, 1778.84525, 1795.4305, 1812.01575, 1828.816917,
        1845.618083, 1862.41925, 1880.13125, 1896.11375, 1912.9705,
        1929.92925, 1946.824208, 1963.719167, 1980.614125, 1997.509083,
        2014.404042, 2031.299, 2048.12575, 2064.897, 2081.6145,
        2098.45285, 2115.2912, 2132.12955, 2148.9679, 2165.80625,
        2182.6446, 2199.48295, 2216.3213, 2233.15965, 2249.998,
        2266.83635, 2283.6747, 2300.51305, 2317.3514, 2334.18975,
        2350.39725, 2366.68125, 2384.11775, 2400.80775, 2417.49775,
        2434.18775, 2450.87775, 2467.56775, 2484.3245, 2501.08125,
    ],
}


def convert(source, wavelengths, label):
    import numpy as np
    import rasterio
    from rasterio.shutil import copy as raster_copy
    from rasterio.windows import Window

    source = Path(source).resolve()
    output_dir = source.parent / (source.stem + '_wavelengths')
    # A new directory prevents accidental replacement of earlier outputs.
    with rasterio.open(source) as src:
        if src.count != len(wavelengths):
            raise ValueError(
                f'{label}: image has {src.count} bands; calibration has '
                f'{len(wavelengths)}. Use the original, unmodified cube.'
            )
        output_dir.mkdir(exist_ok=False)
        output = output_dir / (source.stem + '.dat')
        print(f'Converting {label}: {source.name}', flush=True)
        # GDAL copies the raster in blocks; this does not read the full cube
        # into a Python array. The source is opened read-only.
        with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
            raster_copy(src, output, driver='ENVI', interleave='bsq')
        with rasterio.open(output, 'r+') as dst:
            dst.update_tags(
                ns='ENVI',
                wavelength='{' + ', '.join(f'{v:.9f}' for v in wavelengths) + '}',
                wavelength_units='Nanometers',
            )
        # Reopen to check persisted metadata and sample pixels in every band.
        with rasterio.open(output) as dst:
            tags = dst.tags(ns='ENVI')
            saved = [float(v) for v in tags['wavelength'].strip('{} ').split(',')]
            if not np.allclose(saved, wavelengths, rtol=0, atol=1e-7):
                raise RuntimeError('Wavelength metadata verification failed.')
            if dst.shape != src.shape or dst.count != src.count or dst.dtypes != src.dtypes:
                raise RuntimeError('Raster dimensions or data type changed.')
            if dst.crs != src.crs or not dst.transform.almost_equals(src.transform):
                raise RuntimeError('Spatial reference verification failed.')
            for row, col in [(0, 0), (src.height // 2, src.width // 2),
                             (src.height - 1, src.width - 1)]:
                w = Window(col, row, 1, 1)
                if not np.array_equal(src.read(window=w), dst.read(window=w), equal_nan=True):
                    raise RuntimeError('Sample pixel verification failed.')
        header = output.with_suffix('.hdr')
        if not header.is_file():
            raise RuntimeError('Expected ENVI header was not created.')
        with (output_dir / 'band_wavelength_nm.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['band_1_based', 'wavelength_nm'])
            writer.writerows(enumerate(wavelengths, 1))
        print(f'Verified. In IDCube, load these TWO files together:\n  {header}\n  {output}', flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vn', type=Path, help='Original 76-band VN TIFF')
    p.add_argument('--sw', type=Path, help='Original 90-band SW TIFF')
    args = p.parse_args()
    table = WAVELENGTHS_NM
    jobs = [(args.vn, 'VNIR'), (args.sw, 'SWIR')]
    if not args.vn and not args.sw:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        jobs = []
        for label in ('VNIR', 'SWIR'):
            name = filedialog.askopenfilename(
                title=f'Select the original {label} TIFF (Cancel to skip)',
                filetypes=[('TIFF image', '*.tif *.tiff'), ('All files', '*.*')],
            )
            if name:
                jobs.append((Path(name), label))
        root.destroy()
    for source, label in jobs:
        if source:
            convert(source, table[label], label)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
