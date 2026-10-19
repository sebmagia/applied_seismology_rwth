#!/usr/bin/env python
"""Download the data of the practicals in advance, e.g. before working offline.

    python download_data.py                    # all four practicals (about 840 MB)
    python download_data.py focal forecasting  # only some of them
    python download_data.py --list             # what is available
    python download_data.py --extras           # also the map data (Cartopy) and the PhaseNet
                                               # weights the catalogue notebook downloads itself

You do not have to run this: every notebook downloads its own data the first
time it runs. The files go next to this script, into the same folders the
notebooks use, so the notebooks find them and skip their download.
The data come from the GitHub releases of sebmagia/applied_seismology_rwth.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/sebmagia/applied_seismology_rwth/releases/download"

# practical -> (release tag, data folder next to the notebook, files of the release, approx. size)
#   ("zip", <file>, <folder the zip contains>): extracted into the data folder
#   ("file", <file>, None): copied into the data folder as it is
PRACTICALS = {
    "ambient_noise": ("ambient-noise-practical-v1", "ambient_noise_data", [
        ("zip", "KO.MDNY_noise.zip", "KO.MDNY.00"),
        ("zip", "KO.SLVT_noise.zip", "KO.SLVT.00"),
        ("zip", "KO.BGKT_noise.zip", "KO.BGKT.00"),
        ("zip", "waveforms_foreshocks.zip", "waveforms_foreshocks"),
        ("file", "marmara_coastline.csv", None),
    ], "720 MB", "practical_1_ambient_noise.ipynb"),
    "catalog": ("marmara-catalog-practical-v1", "marmara_catalog_data", [
        ("zip", "data.zip", "data"),
        ("zip", "models.zip", "models"),
    ], "94 MB", "practical_2_earthquake_catalog.ipynb"),
    "focal": ("focal-mechanisms-stress-practical-v1", "focal_stress_data", [
        ("zip", "data.zip", "data"),
        ("zip", "StressInverse.zip", "StressInverse"),
    ], "20 MB", "practical_3_focal_mechanisms_and_stress.ipynb"),
    "forecasting": ("forecasting-practical-v1", "forecasting_practical_data", [
        ("zip", "data.zip", "data"),
    ], "8 MB", "practical_4_aftershock_forecasting.ipynb"),
}


def download(url, target):
    """Download url to target, with a progress line; a partial download never keeps the final name."""
    part = target.with_name(target.name + ".part")

    def progress(blocks, block_size, total):
        done = blocks * block_size / 1e6
        if total > 0:
            print(f"\r    {min(done, total / 1e6):7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, part, reporthook=progress)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    print()
    part.replace(target)


def is_complete(folder, name):
    """True if a data folder is already there (station folders: 7 daily files and the StationXML)."""
    path = folder / name
    if name.endswith(".00"):  # station folder of the ambient-noise practical
        return len(list(path.glob("*.mseed"))) == 7 and len(list(path.glob("*.xml"))) >= 1
    return path.exists()


def get_practical(key):
    tag, folder_name, files, size, notebook = PRACTICALS[key]
    folder = HERE / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n{key}  ({notebook}, about {size}) -> {folder}")
    for kind, filename, content in files:
        url = f"{REPO}/{tag}/{filename}"
        if kind == "file":
            if (folder / filename).exists():
                print(f"  {filename}: already there")
                continue
            print(f"  {filename}: downloading")
            download(url, folder / filename)
            continue

        if is_complete(folder, content):
            print(f"  {content}/: already there")
            continue
        zip_path = folder / filename
        print(f"  {filename}: downloading")
        download(url, zip_path)
        print(f"  {filename}: extracting")
        tmp = folder / f".{content}_tmp"  # extract aside first: an interrupted run leaves no half folder
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        if not (tmp / content).is_dir():
            raise RuntimeError(f"{filename} does not contain the folder {content}/")
        shutil.rmtree(folder / content, ignore_errors=True)  # an incomplete earlier copy
        (tmp / content).rename(folder / content)
        shutil.rmtree(tmp, ignore_errors=True)
        zip_path.unlink(missing_ok=True)
        if not is_complete(folder, content):
            raise RuntimeError(f"{folder / content} is incomplete after extraction")
        print(f"  {content}/: ready")


def get_extras():
    """Map data (Natural Earth, used by Cartopy) and the ETHZ PhaseNet weights; needs the course environment."""
    print("\nextras: map data for Cartopy and PhaseNet weights (needs the course environment)")
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError:
        print("  Cartopy is not installed here: activate the course environment first (see README.md)")
        return False
    for resolution in ("10m", "50m"):
        for category, name in (("physical", "coastline"), ("physical", "land"), ("physical", "ocean"),
                               ("cultural", "admin_0_boundary_lines_land")):
            path = shpreader.natural_earth(resolution=resolution, category=category, name=name)
            print(f"  Natural Earth {resolution} {name}: {path}")
    try:
        import seisbench.models as sbm
    except ImportError:
        print("  SeisBench is not installed here: activate the course environment first (see README.md)")
        return False
    sbm.PhaseNet.from_pretrained("ethz")
    print("  PhaseNet 'ethz' weights: cached by SeisBench")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("practicals", nargs="*", metavar="practical",
                        help=f"one or more of: {', '.join(PRACTICALS)} (default: all)")
    parser.add_argument("--list", action="store_true", help="list the practicals and their data, then stop")
    parser.add_argument("--extras", action="store_true",
                        help="also download the Cartopy map data and the ETHZ PhaseNet weights")
    args = parser.parse_args()
    unknown = [p for p in args.practicals if p not in PRACTICALS]
    if unknown:
        parser.error(f"unknown practical {', '.join(unknown)}; choose from {', '.join(PRACTICALS)}")

    if args.list:
        for key, (tag, folder, files, size, notebook) in PRACTICALS.items():
            print(f"{key:14s} {size:>7s}  {notebook}  ->  {folder}/  (release {tag})")
        return 0

    for key in args.practicals or PRACTICALS:
        try:
            get_practical(key)
        except Exception as exc:
            print(f"\nERROR while downloading '{key}': {exc}\nRun the script again to retry.")
            return 1
    if args.extras and not get_extras():
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
