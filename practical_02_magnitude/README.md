# Practical 2: Earthquake magnitude estimation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebmagia/applied_seismology_rwth/blob/main/practical_02_magnitude/Practical_2_Solution_refined.ipynb)

Worked solution for waveform processing and local magnitude estimation using Sea of Marmara data.

## Run in Colab

1. Click **Open in Colab** and sign in to Google.
2. Run the first code cell to install dependencies and fetch the lesson data.
3. Run the remaining cells from top to bottom. A CPU runtime is sufficient.
4. Save a copy in Drive to retain your edits; download any results you want to keep.

The setup reuses its downloaded repository within a session. To fetch later course updates, start a fresh Colab runtime. Cartopy downloads Natural Earth map data on first use.

## Run locally

From this directory, install `requirements.txt` into your preferred Python environment:

```bash
python -m pip install -r requirements.txt
```

Open `Practical_2_Solution_refined.ipynb` in Jupyter and run cells in order.

## Included files

- `catalog/`: event catalog and associated phase picks.
- `waveforms/`: example MiniSEED recordings.
- `additional_data/`: station response metadata, station coordinates, map layers and attenuation tables.

References for the magnitude scales are included in the notebook. This packaging does not assign a new license to the supplied data or teaching material.

## Preparation notes

- Added a Colab setup cell and removed the unused GDAL import.
- Fixed the multi-station loop to obtain arrival times for the current station, rather than reusing the example station's arrivals.
- Corrected the single-station dataframe selection to use its own boolean mask.
- Cleared saved outputs and execution counts to avoid displaying results computed with the original loop.
- Omitted notebook checkpoints, the unused Ppicks/Spicks tables, unused rawa waveform variants, and unused auxiliary mapping/preparation files.
- Magnitude formulas and other scientific methods are retained from the supplied notebook; this is not an independent scientific review.

Only station XML files matching the included waveforms are retained. The optional EV03532 recording contains TU.VIZD, whose response XML was absent from the supplied archive; the default exercise uses EV01593.

Response-removal errors now stop processing for the affected station; the multi-station loop skips it and reports the reason instead of calculating a magnitude from uncorrected counts. StationXML is matched by network and station.

## Validation

All Python code cells, including maps, ran locally with the supplied default event (1593). The loop produced 22 station estimates and reported response failures for KNAL00 and VIZE00. The later manual station exclusions remain in place. The Colab-only clone/install branch has not yet been tested in Colab; test with Runtime → Run all after uploading. Dependencies are unpinned, so future package updates may require retesting. The badge assumes your default branch is `main`.
