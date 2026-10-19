#!/usr/bin/env python
"""Check that the course environment is complete and works.

    conda activate applied-seismology
    python check_environment.py

The script imports every package the four notebooks use, compares the versions
with environment.yml, checks the installed packages for conflicts (pip check)
and runs a few small tests with synthetic data: ObsPy, a Cartopy map, a b-value
with SeismoStats, a PhaseNet forward pass, a synthetic earthquake associated by
PyOcto and GaMMA, and a synthetic focal mechanism inverted by SKHASH.
It takes a minute or two and needs no internet. Exit code 0 means that
everything works.
"""

import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

USER_MPLBACKEND = os.environ.get("MPLBACKEND", "")  # a value set in the shell is inherited by Jupyter
os.environ["MPLBACKEND"] = "Agg"  # draw figures in memory, no windows

# (distribution name, module to import, expected version prefix; None = any version)
PACKAGES = [
    ("numpy", "numpy", "1.26."),
    ("scipy", "scipy", "1.16."),
    ("pandas", "pandas", "2.2."),
    ("matplotlib", "matplotlib", "3.10."),
    ("statsmodels", "statsmodels", None),
    ("scikit-learn", "sklearn", "1.7.0"),
    ("numba", "numba", None),
    ("obspy", "obspy", "1.5.1"),
    ("cartopy", "cartopy", "0.26.0"),
    ("shapely", "shapely", None),
    ("pyproj", "pyproj", None),
    ("geopandas", "geopandas", None),
    ("torch", "torch", None),
    ("seisbench", "seisbench", "0.12.6"),
    ("pyocto", "pyocto", "0.2.0"),
    ("GMMA", "gamma", "1.2.17"),
    ("SKHASH", "SKHASH", "1.1.5"),
    ("SeismoStats", "seismostats", "1.0.1"),
    ("ipywidgets", "ipywidgets", "8."),
    ("ipympl", "ipympl", "0.9.8"),
    ("ipykernel", "ipykernel", None),
    ("jupyterlab", "jupyterlab", "4."),
]

problems = []           # something is broken
notes = []              # works, but differs from environment.yml
hints = []              # how to fix the problems found
missing_modules = set()  # packages reported as missing (their tests are skipped)

ENV_NAME = "applied-seismology"
IN_CONDA_ENV = (Path(sys.prefix) / "conda-meta").is_dir()
CXX_HINT = f"""\
The C++ libraries of the environment (Matplotlib, PROJ, PyTorch, ...) were built with a newer
compiler than your Linux system, so they need the environment's own copy of the C++ runtime
(libstdc++). This Python loaded the older system copy instead (see "C++ runtime" above). This
usually comes from LD_PRELOAD or LD_LIBRARY_PATH being set in your shell (e.g. in ~/.bashrc).
Switch them off for this environment only:

    conda env config vars set -n {ENV_NAME} LD_PRELOAD= LD_LIBRARY_PATH=
    conda deactivate
    conda activate {ENV_NAME}

and run this check again. conda restores your own values when you deactivate the environment.
The same problem makes "jupyter lab" fail with "No module named 'pysqlite2'"."""


def header(title):
    print(f"\n{title}\n{'-' * len(title)}")


def describe(exc):
    """One-line description of an exception; libstdc++ symbol-version errors are shortened."""
    msg = f"{type(exc).__name__}: {exc}"
    m = re.search(r"version `([A-Z]+_[0-9.]+)' not found", msg)
    if m and "libstdc++" in msg:
        return f"{type(exc).__name__}: the C++ runtime that was loaded lacks {m.group(1)} (see 'C++ runtime')"
    return msg


# ---------------------------------------------------------------------------------------------
# 1. Python and packages
# ---------------------------------------------------------------------------------------------
def check_python():
    header("Python")
    print(f"  Python   {platform.python_version()}  ({sys.executable})")
    print(f"  System   {platform.platform()}")
    if sys.platform.startswith("linux"):  # settings that change which shared libraries are loaded
        for var in ("LD_PRELOAD", "LD_LIBRARY_PATH"):
            if os.environ.get(var):
                print(f"  {var:15s} {os.environ[var]}")
        if Path("/etc/ld.so.preload").is_file():
            print(f"  /etc/ld.so.preload: {Path('/etc/ld.so.preload').read_text().strip() or '(empty)'}")
    if USER_MPLBACKEND:
        print(f"  MPLBACKEND      {USER_MPLBACKEND}")
        if "inline" not in USER_MPLBACKEND and "ipympl" not in USER_MPLBACKEND:
            notes.append(f"MPLBACKEND={USER_MPLBACKEND} is set: Jupyter kernels inherit it. The course notebooks choose "
                         f"their own backend, but other notebooks would open figures in separate windows "
                         f"(the conda environment switches it off: conda env update -n {ENV_NAME} -f environment.yml)")
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {platform.python_version()}: the course environment uses Python 3.12")
        hints.append(f"Activate the course environment first:  conda activate {ENV_NAME}   (see README.md)")


def check_packages():
    header("Packages")
    for dist, module, expected in PACKAGES:
        try:
            installed = version(dist)
        except PackageNotFoundError:
            installed = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import_module(module)
            error = None
        except Exception as exc:  # also DLL / shared-library errors
            error = describe(exc)
            if "pysqlite2" in error:  # Jupyter's fallback import hides the real error
                error += " (Python's sqlite3 module failed to load: see the sqlite3 test below)"
        if installed is None and error:
            status = "MISSING"
            problems.append(f"{dist} is not installed")
            missing_modules.add(module)
            if not any("conda activate" in h for h in hints):
                hints.append(f"Is the course environment active? Run:  conda activate {ENV_NAME}   (see README.md)")
        elif error:
            status = "IMPORT ERROR"
            problems.append(f"{dist} {installed} cannot be imported: {error}")
        elif expected and not str(installed).startswith(expected):
            status = f"ok, but environment.yml has {expected.rstrip('.')}"
            notes.append(f"{dist} {installed} instead of {expected.rstrip('.')}")
        else:
            status = "ok"
        print(f"  {dist:13s} {installed or '-':14s} {status}")


def check_cxx_runtime():
    """Linux + conda: the environment's C++ libraries need the environment's own libstdc++, not the system's."""
    if not (sys.platform.startswith("linux") and IN_CONDA_ENV):
        return
    header("C++ runtime (libstdc++)")
    own = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    try:
        with open("/proc/self/maps") as f:  # the shared libraries this Python process has loaded
            loaded = sorted({line.split(maxsplit=5)[-1].strip() for line in f if "libstdc++.so" in line})
    except OSError:
        loaded = []
    prefix = Path(sys.prefix).resolve()
    foreign = [p for p in loaded if not Path(p.replace(" (deleted)", "")).resolve().is_relative_to(prefix)]
    print(f"  environment's copy  {own if own.exists() else 'MISSING'}")
    print(f"  loaded by Python    {', '.join(loaded) or '(none yet)'}")
    if not own.exists():
        problems.append("the environment has no libstdc++ of its own")
        hints.append(f"Install the environment's C++ runtime:  conda install -n {ENV_NAME} -c conda-forge libstdcxx libgcc")
    elif foreign:
        problems.append(f"Python loaded the system's libstdc++ ({foreign[0]}) instead of the environment's")
        hints.append(CXX_HINT)


def check_pip():
    header("Conflicts between installed packages (pip check)")
    res = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    out = (res.stdout + res.stderr).strip()
    print("  " + out.replace("\n", "\n  "))
    if res.returncode != 0:
        problems.append("pip check found packages with unmet or conflicting requirements (see above)")


# ---------------------------------------------------------------------------------------------
# 2. Small tests with synthetic data
# ---------------------------------------------------------------------------------------------
def test_sqlite():
    import sqlite3

    con = sqlite3.connect(":memory:")
    con.execute("create table t (x integer)")
    con.close()
    return "Python's sqlite3 module works (Jupyter needs it)"


def test_obspy():
    from obspy import read
    from obspy.imaging.beachball import beach  # noqa: F401  (used by the focal-mechanism notebook)
    from obspy.signal import PPSD  # noqa: F401  (used by the ambient-noise notebook)

    st = read()  # ObsPy's bundled example data, no download
    st.detrend("demean")
    st.filter("bandpass", freqmin=1.0, freqmax=10.0)
    return f"read and filtered {len(st)} example traces"


def test_maps():
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([27.5, 29.5, 40.3, 41.4], crs=ccrs.PlateCarree())
    ax.plot([28.2], [40.9], "r*", transform=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    fig.canvas.draw()
    plt.close(fig)
    return "drew a Cartopy map (the notebooks download the coastlines on first use)"


def test_seismostats():
    import numpy as np
    from seismostats import Catalog  # noqa: F401
    from seismostats.analysis import ClassicBValueEstimator

    rng = np.random.default_rng(42)
    # Gutenberg-Richter magnitudes with b = 1 above Mc = 1.0, in bins of 0.1
    mags = np.round(0.95 + 1e-6 + rng.exponential(1 / np.log(10), 20000), 1)
    b = ClassicBValueEstimator().calculate(mags, mc=1.0, delta_m=0.1)
    if not 0.9 < b < 1.1:
        raise RuntimeError(f"b-value of a synthetic catalogue with b = 1 came out as {b:.2f}")
    return f"b-value of a synthetic catalogue: {b:.2f} (true value 1.00)"


def test_phasenet():
    import seisbench.models as sbm
    import torch

    model = sbm.PhaseNet()  # untrained network: tests PyTorch without downloading weights
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 3001))
    if tuple(out.shape) != (1, 3, 3001):
        raise RuntimeError(f"unexpected PhaseNet output shape {tuple(out.shape)}")
    return f"PhaseNet forward pass on the CPU (torch {torch.__version__})"


def synthetic_picks():
    """Six stations of the catalogue practical and the P and S arrivals of one synthetic earthquake."""
    import numpy as np
    import pandas as pd
    import pyocto

    stations = pd.DataFrame({
        "id": ["KO.ARMT.", "KO.BGKT.", "KO.ISK.", "KO.MDNY.", "KO.SLVT.", "TU.ELBA."],
        "latitude": [40.5683, 41.1810, 41.0615, 40.3709, 41.2288, 41.1469],
        "longitude": [28.8660, 28.7730, 29.0592, 28.8846, 28.2098, 28.4307],
        "elevation": [320.0, 80.0, 85.0, 116.0, 180.0, 331.0],
    })
    velocity_model = pyocto.VelocityModel0D(p_velocity=6.0, s_velocity=3.5, tolerance=1.5,
                                            association_cutoff_distance=150.0)
    associator = pyocto.OctoAssociator.from_area(
        lat=(40.6, 41.05), lon=(27.7, 28.9), zlim=(0, 40), velocity_model=velocity_model,
        min_node_size=5.0, min_node_size_location=0.2, location_split_depth=12, location_split_return=10,
        time_before=90, n_picks=8, n_p_picks=3, n_s_picks=3, n_p_and_s_picks=3,
        pick_match_tolerance=1.5, n_threads=2)
    associator.transform_stations(stations)

    # the earthquake: 40.88 N, 28.22 E, 10 km deep (elevation -10 km gives z = +10 km)
    source = pd.DataFrame({"latitude": [40.88], "longitude": [28.22], "elevation": [-10000.0]})
    associator.transform_stations(source)
    x0, y0, z0 = source.loc[0, ["x", "y", "z"]]
    t0 = 1569312022.0  # 2019-09-24 08:00:22 UTC

    rows = []
    for st in stations.itertuples():
        dist = np.sqrt((st.x - x0) ** 2 + (st.y - y0) ** 2 + (st.z - z0) ** 2)
        rows.append({"station": st.id, "time": t0 + dist / 6.0, "phase": "P", "probability": 0.9})
        rows.append({"station": st.id, "time": t0 + dist / 3.5, "phase": "S", "probability": 0.9})
    picks = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return associator, stations, picks, (40.88, 28.22, 10.0, t0)


def test_pyocto():
    from obspy.geodetics import gps2dist_azimuth

    associator, stations, picks, (lat, lon, depth, t0) = synthetic_picks()
    events, assignments = associator.associate(picks, stations)
    associator.transform_events(events)
    if len(events) != 1:
        notes.append(f"PyOcto found {len(events)} events in the synthetic test (expected 1)")
        return f"ran, but found {len(events)} events instead of 1"
    ev = events.iloc[0]
    err = gps2dist_azimuth(lat, lon, ev.latitude, ev.longitude)[0] / 1000
    return (f"associated {len(assignments)} of {len(picks)} synthetic picks into 1 event, "
            f"{err:.1f} km from the true epicentre")


def test_gamma():
    import numpy as np
    import pandas as pd
    from gamma.utils import association, estimate_eps

    associator, stations, picks, _ = synthetic_picks()
    gamma_picks = pd.DataFrame({
        "id": picks.station,
        "timestamp": pd.to_datetime(picks.time, unit="s", utc=True).dt.tz_localize(None),
        "type": picks.phase.str.lower(), "prob": picks.probability})
    gamma_stations = stations[["id", "x", "y", "z"]].rename(columns={"x": "x(km)", "y": "y(km)", "z": "z(km)"})
    config = {
        "dims": ["x(km)", "y(km)", "z(km)"],
        "x(km)": tuple(associator.xlim), "y(km)": tuple(associator.ylim), "z(km)": (0, 40),
        "vel": {"p": 6.0, "s": 3.5}, "use_amplitude": False,
        "use_dbscan": True, "dbscan_eps": estimate_eps(gamma_stations, vp=6.0), "dbscan_min_samples": 3,
        "oversample_factor": 4, "covariance_prior": [1.0, 1.0],
        "min_picks_per_eq": 8, "min_p_picks_per_eq": 3, "min_s_picks_per_eq": 3,
        "max_sigma11": 1.5, "ncpu": 1}
    config["bfgs_bounds"] = (config["x(km)"], config["y(km)"], (0, 40), (None, None))
    np.random.seed(42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, links = association(gamma_picks, gamma_stations, config, method="BGMM")
    if len(events) != 1:
        notes.append(f"GaMMA found {len(events)} events in the synthetic test (expected 1)")
        return f"ran (numba works), but found {len(events)} events instead of 1"
    return f"associated {len(links)} of {len(picks)} synthetic picks into 1 event (numba works)"


# station geometry of event 2815 from the focal-mechanism practical:
# (station, network, channel, take-off, take-off uncertainty, azimuth, azimuth uncertainty)
SKHASH_STATIONS = [
    ("ELBA", "TU", "HHZ", 81.4, 6.5, 26.2, 1.9), ("CTKS", "KO", "HHZ", 86.3, 6.1, 27.7, 1.4),
    ("BIRL", "TU", "HHZ", 87.3, 5.1, 74.2, 1.3), ("CATL", "TU", "HHZ", 87.9, 4.5, 10.4, 1.3),
    ("BAND", "TU", "HHZ", 87.9, 4.5, 192.0, 1.3), ("BGKT", "KO", "HHZ", 88.4, 5.1, 50.5, 1.2),
    ("TKR", "KO", "HHZ", 92.0, 5.6, 285.1, 1.2), ("ARMT", "KO", "HHZ", 92.0, 5.5, 120.0, 1.1),
    ("EDC", "KO", "HHZ", 92.0, 4.9, 209.7, 1.1), ("MADA", "TU", "HHZ", 92.0, 4.8, 243.7, 1.1),
    ("KCTX", "KO", "HHZ", 92.0, 4.6, 172.6, 1.1), ("EYUP", "TU", "HHZ", 92.0, 4.3, 62.9, 1.1),
    ("KNAL", "PZ", "EHZ", 92.0, 3.9, 84.6, 1.0), ("CTYL", "KO", "HHZ", 92.0, 3.7, 3.7, 1.0),
    ("KAVV", "KO", "HHZ", 92.0, 3.1, 70.7, 1.0), ("MRTI", "PZ", "HHZ", 92.0, 3.1, 92.1, 1.0),
    ("MDNY", "KO", "HHZ", 93.3, 2.4, 133.7, 1.0), ("NMR2", "KO", "HHZ", 93.3, 3.0, 259.8, 1.0),
    ("KLYT", "KO", "HHZ", 94.2, 2.2, 56.2, 1.0), ("VIZE", "TU", "HHZ", 94.9, 2.3, 337.9, 1.0),
    ("SATE", "TU", "HHZ", 95.5, 2.3, 77.9, 1.0), ("TKOY", "TU", "HHZ", 95.5, 2.3, 258.2, 1.0),
    ("GEML", "KO", "HHZ", 95.5, 2.2, 117.0, 1.0), ("KRBG", "KO", "HHZ", 96.1, 2.0, 237.9, 1.0),
    ("CMH", "TU", "HHZ", 96.1, 2.0, 193.7, 1.0), ("YLV", "KO", "HHZ", 97.1, 2.2, 107.7, 1.0),
    ("ATI", "TU", "HHZ", 97.1, 2.2, 214.1, 1.0), ("TMAL", "TU", "HHZ", 97.5, 1.7, 276.5, 1.0),
    ("ERE", "TU", "HHZ", 97.5, 1.7, 147.8, 1.0), ("ULDT", "TU", "HHZ", 98.0, 2.0, 135.5, 1.0),
    ("ISIL", "TU", "HHZ", 102.0, 3.1, 73.1, 1.0), ("AVCA", "TU", "HHZ", 102.0, 3.1, 344.7, 1.0),
    ("HRTX", "KO", "HHZ", 102.0, 3.1, 92.0, 1.0), ("SILT", "KO", "HHZ", 102.0, 26.9, 73.6, 1.0),
    ("BKES", "TU", "HHZ", 102.0, 26.9, 192.0, 1.0), ("NEV", "TU", "HHZ", 129.6, 27.3, 220.1, 1.0),
    ("ARMN", "TU", "HHZ", 129.6, 25.1, 329.4, 1.0), ("BALY", "TU", "HHZ", 129.6, 25.1, 203.3, 1.0),
    ("ADVT", "KO", "HHZ", 129.6, 1.0, 109.4, 1.0), ("LAPS", "TU", "HHZ", 129.6, 1.0, 241.3, 1.0),
    ("LAP", "KO", "HHZ", 129.6, 1.0, 247.5, 1.0), ("GOKT", "TU", "HHZ", 129.6, 1.0, 261.9, 1.0),
    ("DURS", "TU", "HHZ", 129.6, 1.0, 171.6, 1.0), ("KIRK", "TU", "HHZ", 129.6, 1.0, 317.8, 1.0),
    ("ERIK", "KO", "HHZ", 129.6, 1.0, 262.9, 1.0),
]

SKHASH_CONTROL = """$input_format
skhash
$catfile
eq_catalog.csv
$fpfile
pol.csv
$outfile1
out_check_mechanism.csv
$outfile2
out_check_acceptable_planes.csv
$outfile_pol_agree
out_check_pol_agree.csv
$outfile_pol_info
out_check_pol_info.csv
$npolmin
6
$max_agap
180
$max_pgap
90
$badfrac
0.1
$badmin
2
$nmc
30
$maxout
500
$dang
5
"""


def _fault_vectors(strike, dip, rake):
    """Fault normal and slip vector (north, east, down), as in the focal-mechanism notebook."""
    import numpy as np
    s, d, r = np.radians([strike, dip, rake])
    n = np.array([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)])
    u = np.array([np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                  np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s), -np.sin(r) * np.sin(d)])
    return n, u


def _kagan_angle(sdr1, sdr2):
    import numpy as np

    def axes(sdr):
        n, u = _fault_vectors(*sdr)
        t, p = (n + u) / np.sqrt(2), (n - u) / np.sqrt(2)
        return np.column_stack([t, p, np.cross(t, p)])

    a1, a2 = axes(sdr1), axes(sdr2)
    sym = [np.diag(v) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    return min(np.degrees(np.arccos(np.clip((np.trace(a1 @ s @ a2.T) - 1) / 2, -1, 1))) for s in sym)


def test_skhash():
    import numpy as np
    import pandas as pd
    import SKHASH  # noqa: F401  (SKHASH itself runs in a separate process below, like in the notebook)

    true_sdr = (30.0, 50.0, 60.0)  # an oblique mechanism, as in the take-off convention check of the notebook
    n, u = _fault_vectors(*true_sdr)
    lines = ["event_id, station, network, location, channel, p_polarity, takeoff, takeoff_uncertainty, "
             "azimuth, azimuth_uncertainty"]
    for sta, net, cha, tk, tku, az, azu in SKHASH_STATIONS:
        i, a = np.radians(180 - tk), np.radians(az)  # SKHASH take-off: 0 = up
        g = np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])
        rad = 2 * (g @ n) * (g @ u)
        pol = 0.0 if abs(rad) < 0.1 else float(np.sign(rad))
        lines.append(f"2815, {sta}, {net}, --, {cha}, {pol}, {tk}, {tku}, {az}, {azu}")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "eq_catalog.csv").write_text(
            "time, latitude, longitude, depth, horz_uncert_km, vert_uncert_km, mag, event_id\n"
            "2025-04-23 09:13:06.544, 40.847738, 28.235647, 10.439, 0.618, 2.513, 3.79, 2815\n")
        (folder / "pol.csv").write_text("\n".join(lines) + "\n")
        (folder / "control_file.txt").write_text(SKHASH_CONTROL)
        # run SKHASH exactly like run_skhash() in the notebook
        code = "import sys; from SKHASH.SKHASH import main; sys.argv = ['SKHASH', 'control_file.txt']; sys.exit(main())"
        res = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True, text=True, timeout=900)
        out = folder / "out_check_mechanism.csv"
        if not out.exists() or len(pd.read_csv(out)) == 0:
            tail = (res.stderr.strip() or res.stdout.strip()).splitlines()[-5:]
            raise RuntimeError("SKHASH produced no mechanism. Last lines of its output:\n      " + "\n      ".join(tail))
        m = pd.read_csv(out).iloc[0]
    kagan = _kagan_angle((m.strike, m.dip, m.rake), true_sdr)
    if kagan > 30:
        notes.append(f"SKHASH recovered the synthetic mechanism only within {kagan:.0f} degrees")
    return (f"inverted synthetic polarities at {len(SKHASH_STATIONS)} stations: {m.strike:.0f}/{m.dip:.0f}/{m.rake:.0f} "
            f"(quality {m.quality}), {kagan:.0f} degrees from the true 30/50/60")


def test_widgets():
    import ipywidgets as widgets
    widgets.VBox([widgets.HTML("x"), widgets.HBox([widgets.Button(description="up")]), widgets.Output()])
    return "created the widgets of the polarity picker"


def test_ipympl():
    import ipympl  # noqa: F401

    # draw with the backend of %matplotlib widget, in a separate process (this one uses Agg)
    code = ("import matplotlib.pyplot as plt; fig, ax = plt.subplots(); ax.plot([1, 2]); fig.canvas.draw(); "
            "print(type(fig.canvas).__module__)")
    env = {**os.environ, "MPLBACKEND": "module://ipympl.backend_nbagg"}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    if res.returncode != 0 or "ipympl" not in res.stdout:
        tail = (res.stderr.strip() or res.stdout.strip()).splitlines()[-3:]
        raise RuntimeError("the ipympl backend could not draw a figure:\n      " + "\n      ".join(tail))
    return "drew a figure with the backend of %matplotlib widget (interactive figures)"


TESTS = [
    ("sqlite3 (for Jupyter)", test_sqlite),
    ("ObsPy", test_obspy),
    ("Cartopy map", test_maps),
    ("SeismoStats b-value", test_seismostats),
    ("SeisBench PhaseNet", test_phasenet),
    ("PyOcto association", test_pyocto),
    ("GaMMA association", test_gamma),
    ("SKHASH focal mechanism", test_skhash),
    ("ipywidgets", test_widgets),
    ("ipympl figures", test_ipympl),
]


def run_tests():
    header("Tests with synthetic data")
    for name, func in TESTS:
        t = time.perf_counter()
        try:
            msg = func()
            print(f"  {name:24s} ok      {msg}  [{time.perf_counter() - t:.0f} s]")
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] in missing_modules:  # already reported in the package list
                print(f"  {name:24s} skipped ({exc.name} is not installed)")
                continue
            print(f"  {name:24s} FAILED  {exc}")
            problems.append(f"test '{name}' failed: {exc}")
        except Exception as exc:
            print(f"  {name:24s} FAILED  {type(exc).__name__}: {exc}")
            where = traceback.extract_tb(exc.__traceback__)[-1]
            print(f"      (raised in {where.filename}, line {where.lineno})")
            problems.append(f"test '{name}' failed: {describe(exc).splitlines()[0]}")


def main():
    print("Checking the course environment (this takes a minute or two) ...")
    check_python()
    check_packages()
    check_cxx_runtime()
    check_pip()
    run_tests()
    header("Result")
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print(f"  {len(problems)} problem(s):")
        for p in problems:
            print(f"   - {p}")
        if any(("GLIBCXX_" in p or "CXXABI_" in p) for p in problems) and CXX_HINT not in hints:  # loaded copy is the env's, but too old
            hints.append(f"Some libraries need a newer C++ runtime: conda update -n {ENV_NAME} -c conda-forge libstdcxx libgcc")
        if hints:
            print("\n  How to fix it:")
            for h in hints:
                print(textwrap.indent(h, "    "))
        print("\n  If the problem stays, send the complete output of this check to your instructor.")
        return 1
    print("  All checks passed: the environment is ready for the four notebooks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
