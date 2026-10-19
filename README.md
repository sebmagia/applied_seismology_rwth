# Applied Seismology practicals – running them on your own computer

Four Jupyter notebooks from the Applied Seismology course (RWTH Aachen), with one Python
environment that contains everything they need.

| Practical | Notebook | Topic | Data (downloaded on first run) |
|---|---|---|---|
| 1 | `practical_1_ambient_noise.ipynb` | Ambient seismic noise and station quality control: miniSEED, StationXML, PPSD, daily and weekly noise cycles (KOERI stations, Sea of Marmara) | ~720 MB |
| 2 | `practical_2_earthquake_catalog.ipynb` | Build an earthquake catalog: PhaseNet picks (SeisBench), PyOcto and GaMMA association, local magnitudes | ~95 MB |
| 3 | `practical_3_focal_mechanisms_and_stress.ipynb` | Focal mechanisms with SKHASH (interactive polarity picking) and stress inversion with StressInverse | ~20 MB |
| 4 | `practical_4_aftershock_forecasting.ipynb` | Gutenberg–Richter law, b-value and aftershock forecasts (SeismoStats) | 8 MB |

All notebooks have the same structure: *Prepare the workspace* (environment check, imports, data download),
then **Part 1, Part 2, …** with numbered sections (1.1, 1.2, …) and **Exercises** numbered through the notebook.

The data come from the [GitHub releases](https://github.com/sebmagia/applied_seismology_rwth/releases)
of the course repository.

Earlier practicals of this repository (magnitude estimation with machine learning, local magnitudes) are kept in
its history: [commit `d680e30`](https://github.com/sebmagia/applied_seismology_rwth/tree/d680e30b98ca1d87e12dc6186df29b05bebedf34).

## 1. Install the environment (once)

**a. Install Miniforge** (a small conda installer that uses the free conda-forge packages):

- **Windows:** download and run `Miniforge3-Windows-x86_64.exe` from <https://conda-forge.org/download/>
  ("Just Me" is fine). Afterwards, open **Miniforge Prompt** from the Start menu for all commands below.
- **macOS / Linux:** in a terminal:

  ```bash
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3-$(uname)-$(uname -m).sh
  ```

  then close and reopen the terminal.

If you already have Anaconda or Miniconda you can use it instead; the environment file only uses conda-forge.

**b. Create the environment.** Go to this folder and create it from `environment.yml`
(this takes a few minutes and about 3–4 GB of disk space):

```bash
cd path/to/applied_seismology_practicals
conda env create -f environment.yml
```

**c. Check it:**

```bash
conda activate applied-seismology
python check_environment.py
```

It imports every package, looks for version conflicts and runs a small test of each tool (ObsPy, Cartopy,
SeismoStats, PhaseNet, PyOcto, GaMMA, SKHASH). It takes a minute or two and ends with `All checks passed`
when everything works.

## 2. Start Jupyter (every time)

```bash
conda activate applied-seismology
cd path/to/applied_seismology_practicals
jupyter lab
```

Open a notebook and run the cells from the top (or *Run → Run All Cells*). The first cell checks that the
notebook runs in the course environment.

**VS Code** instead of JupyterLab: install the *Python* and *Jupyter* extensions, open this folder, open a
notebook, click *Select Kernel → Python Environments* and choose `applied-seismology`.

**A Jupyter you already use** (installed elsewhere): register the environment as a kernel once, then pick
*Python (applied-seismology)* in the notebook's kernel menu:

```bash
conda activate applied-seismology
python -m ipykernel install --user --name applied-seismology --display-name "Python (applied-seismology)"
```

## 3. The data

Each notebook downloads its own data the first time you run it (internet needed) into a folder next to it:
`ambient_noise_data/`, `marmara_catalog_data/`, `focal_stress_data/` and `forecasting_practical_data/`.
From then on it finds them there and works offline.

To download everything in advance (about 840 MB), e.g. before working without internet:

```bash
python download_data.py              # all four practicals
python download_data.py focal        # or only some: ambient_noise, catalog, focal, forecasting
python download_data.py --extras     # also the coastlines for the maps and the 'ethz' PhaseNet weights
```

`--extras` needs the activated environment. Without it, the maps of the focal-mechanism notebook and the
`ethz` weights of the catalog notebook are downloaded the first time they are used.

To start a practical again from scratch (e.g. your polarity picks of the focal-mechanism practical, which are
saved in `focal_stress_data/data/skhash/`), delete its data folder.

## 4. Updating or removing the environment

```bash
conda env update -n applied-seismology -f environment.yml --prune    # after environment.yml changed
conda env remove -n applied-seismology                               # remove it completely
```

## Without conda (pip)

You need **Python 3.12** (64-bit); 3.13 does not work yet (pyocto and numpy 1.26 have no ready-made
packages for it). On Intel Macs use conda instead (ObsPy has no ready-made pip package for them).

```bash
python3.12 -m venv .venv                 # Windows: py -3.12 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# Linux only, so that pip does not download ~3 GB of GPU libraries:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python check_environment.py
jupyter lab
```

With [uv](https://docs.astral.sh/uv/), `uv venv --python 3.12` creates the environment (and downloads
Python 3.12 if needed); then use `uv pip install` instead of `python -m pip install`.

## Troubleshooting

- **The first cell says "Not installed for this kernel"**: Jupyter was started from another Python. Close it,
  run `conda activate applied-seismology` and start `jupyter lab` again (in VS Code: select the
  `applied-seismology` kernel).
- **`conda activate` does not work**: on Windows use the *Miniforge Prompt*; elsewhere run `conda init` once
  and open a new terminal.
- **A download was interrupted**: run the cell (or `download_data.py`) again. If a notebook then says the data
  are "already available" but files are missing, delete that practical's data folder and run it again.
- **The polarity picker shows a line of text like `VBox(...)` instead of buttons**: the notebook is not running
  in the course environment's Jupyter. Start JupyterLab as in section 2, or use
  `pick_polarities(..., buttons=False)` and type the answers.
- **Figures are missing, or show a line of text like `Canvas(...)`**: the interactive figures need ipympl,
  which is part of the course environment; start JupyterLab as in section 2. If you installed with pip, run
  `python -m pip install -r requirements.txt` again (it includes ipympl).
- **An ObsPy waveform plot (`st.plot()`) stays empty**: ObsPy closes its figure after drawing it, which removes
  an interactive figure from the notebook. Pass `handle=True`, as the notebooks do: `st.plot(handle=True);`.
- **Figures open in separate windows, or a cell with a figure never finishes**: your computer sets the
  environment variable `MPLBACKEND` (e.g. to `TkAgg`). The notebooks choose their own figure backend, and the
  conda environment switches the variable off while it is active (`conda env update -f environment.yml` adds
  this to an older installation).
- **`python check_environment.py` reports a problem**: send its complete output to your instructor.

