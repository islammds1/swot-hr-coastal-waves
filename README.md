# swot-hr-waves

Swell wavelength, period and direction from SWOT KaRIn **HR pixel clouds**
(`L2_HR_PIXC`), on 5 km boxes, with two independent estimators (2-D FFT and
Radon / Fourier-slice), validated against **MARC/WW3**, plus the diagnostics
used in the paper (instrument response, along-track bias, bathymetric control).

Pipeline version **v4.3**, revised after Yu et al. (2026), *A first analysis of
ocean swell observations with SWOT's High-Rate mode*, IEEE TGRS (hal-05610282).

---

## 1. Install

```bash
git clone https://github.com/<user>/swot-hr-waves.git
cd swot-hr-waves
conda env create -f environment.yml      # creates env "swothr", installs the package
conda activate swothr
```

Without conda: `pip install -e ".[maps,dev]"` (Python ≥ 3.10; `maps` pulls cartopy,
needed only for the storm map).

## 2. Tell it where the data are (once per machine)

```bash
cp config/paths.example.yaml config/paths.yaml     # git-ignored
nano config/paths.yaml
swothr paths                                       # prints each path + ok/MISSING
```

```yaml
root:       /media/saiful/LaCie/Coastal_Enginnering_SWOT_HR
bathymetry: /media/saiful/LaCie/Data/Bathymetry/E4_2024.nc
storms_dir: /media/saiful/LaCie/SWOT_HR_Project/Storms
```

Expected layout under `root` (nothing here is ever committed):

```
root/
├── Data/
│   ├── SWOT HR/<stem>/SWOT_L2_HR_PIXC_*.nc
│   ├── SWOT_LR/SWOT_L2_LR_SSH_Expert_<cycle>_<pass>_*.nc
│   ├── Surface current/Surface_current_<stem>.nc     # MARC MARS3D
│   └── Wave Validation/MARC_WW3-NORGAS-2M_<stem>.nc
├── Results/{FFT,radon}/new/        # written by the pipelines
└── Outputs/                        # written by the analysis steps
```

## 3. Run

```bash
swothr list                                         # all steps
swothr fft   -c config/storms/nelson_42.yaml        # one storm
swothr radon -c config/storms/*.yaml --keep-going   # every storm in turn
swothr compare                                      # FFT / Radon vs WW3, all storms
swothr bias                                         # origin of the wavelength bias
```

Every step is also a plain module: `python -m swothr.pipelines.fft --config config/storms/nelson_42.yaml`
(handy in Spyder: *Run → Configuration per file → Command line options*).

### Typical order

| # | step | what it does | reads | writes |
|---|------|--------------|-------|--------|
| 1 | `fft` | FFT directional spectrum per 5 km box: λ, T, θ, 180° lobe from Re⟨h σ0*⟩, observability grade A–D | PIXC, LR SSH, bathy, current, WW3 | `Results/FFT/new/swot_5km_FFT_<TAG>.csv`, `.geojson`, maps |
| 2 | `radon` | Same boxes, Radon / Fourier-slice estimator | same | `Results/radon/new/swot_5km_RADON_<TAG>.csv` |
| 3 | `rain-flag` | HR-internal contamination score, tested against LR flags | PIXC, LR SSH | `Outputs/rain_screen_<stem>.csv` |
| 4 | `compare` | Paired per-box FFT − WW3 and Radon − WW3 statistics, all storms | step 1–2 CSVs, WW3 | `Outputs/comparison_ww3/new/fft_radon_ww3_*.csv/png` |
| 5 | `bias` | Wavelength bias vs swath position, depth, k-components, alignment | `fft_radon_ww3_perbox.csv` + step 1–2 CSVs | `fig_wavelength_bias_origin.png/csv` |
| 6 | `mask-sens` | Fig. S3, sensitivity to the power-response mask | FFT CSVs (`sens_m*` columns) | `Figure_S3.jpg` |
| 7 | `synthetic` | Linear-filter centroid bias on synthetic spectra (no data) | – | `Outputs/synthetic/` |
| 8 | `bathy` | Shoaling, refraction ray tracing, Sark diffraction (Ciarán) | FFT CSV, bathymetry | `Results/bathy_relations/` |
| 9 | `box-diag` | Per-box SSH / spectrum / sinogram / cross-spectrum panels | PIXC, CSVs | `Results/box_diagnostics/` |
| – | `storms` | 2×2 gust-footprint and track map | `storms_dir` | `Outputs/storms_comparison_2x2_cartopy.png` |
| – | `validate` | Older single-scene validation on a v4 CSV | v4 CSV, WW3 | `model_validation_*` |
| – | `compare-recomputed-T` | Alternative to `compare`: model periods recomputed from λp | as `compare` | as `compare` |

## 4. Configuration

All settings stay where they were, as UPPER-CASE constants at the top of each
module, with their physical justification in the comments. A run file only
overrides what changes from one scene to the next:

```yaml
# config/storms/nelson_42.yaml
common:   {DATE: "2024-03-29", CYCLE: 13, PASS: 42, UTM_ZONE: 30}
pipeline:                       # shared by fft and radon
  SWOT_FILES: "{root}/Data/SWOT HR/Nelson_42/SWOT_L2_HR_PIXC_*.nc"
  SWOT_QUAL_BITMASK: 0xF908B402
fft:      {TAG: Nelson_42_FFT_v4.3}
radon:    {TAG: Nelson_42_RADON_v4.3}
```

Sections read by each step: `common` + `pipeline` + `fft`/`radon` (pipelines),
`rain_flag`, `compare`, `validate`, `bias`, `mask_sensitivity`, `synthetic`,
`bathy`, `box_diag`, `storms`. Any UPPER-CASE setting of a script can be
overridden (e.g. `CROSS_TRACK_MIN`, `RESPONSE_MIN_MASK`, `GRANULES`). The run
prints which settings it took from the file and warns about keys the script
does not define, so a typo cannot pass silently. New storm: copy
`config/storms/_template.yaml`.

Keep `TAG = <stem>_FFT_v4.3` / `<stem>_RADON_v4.3`: `compare` locates the
pipeline outputs as `swot_5km_{EST}_{stem}_{EST}_v4.3.csv`.

## 5. Repository layout

```
src/swothr/
├── cli.py                  swothr command
├── config.py               paths.yaml + per-run overrides
├── pipelines/   fft.py  radon.py  rain_flag.py
├── analysis/    compare_ww3.py  compare_ww3_recomputed_T.py  along_track_bias.py
│                mask_sensitivity.py  response_synthetic.py  bathymetric_control.py
│                box_diagnostic.py  validate_profiles.py
├── plotting/    storm_tracker.py
└── utils/       response.py (H3 boxcar · sinc response)  stats.py (bootstrap)  plotting.py (paper style)
config/          paths.example.yaml, storms/*.yaml
tests/           pytest, no data needed
```

## 6. Conventions worth knowing

- **Velocity-bunching and tilt signs** are fixed from theory (Yu et al. 2026, §V-B/C):
  `VB_SIGN = TILT_SIGN = −1`. Each FFT run writes a scene-level antisymmetry test of
  Im C (`diag_tilt_antisymmetry_<TAG>.png`); if it fails, every direction is 180° suspect.
- **Unresolved lobes are flagged, never filled** (`LOBE_FALLBACK = "flag"`). Where
  `lobe_source == "model"` the lobe came from WW3, so those boxes are excluded from
  direction validation.
- **WW3 direction** is nautical *from*; `fp` is intrinsic. Model λp comes from `fp`
  through the dispersion relation with the same depth and current as SWOT.
- **Instrument response**: L2 3×3 boxcar `H(k) = (1 + 2 cos kd)/3`, zero at λ = 3d,
  −3 dB in power at λ = 6.44d, with d_al = 22 m and d_ac from the across-track posting.

## 7. Tests

```bash
pytest -q
```

Unit tests cover the response model, bootstrap statistics and the config
mechanism; every step is import-checked. They need no SWOT data and also run on
GitHub Actions at each push.

## Reference

Yu et al. (2026). A first analysis of ocean swell observations with SWOT's
High-Rate mode. *IEEE Transactions on Geoscience and Remote Sensing*. hal-05610282.
