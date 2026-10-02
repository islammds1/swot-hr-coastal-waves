from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =========================================================
# Global font settings
# =========================================================
mpl.rcParams['font.family'] = 'DejaVu Sans'   # swap for 'Arial', 'Calibri', etc. if installed
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['figure.titleweight'] = 'bold'

# =========================================================
# Define storms: file paths + display name
# =========================================================
from swothr.config import path, apply_overrides

STORMS_DIR = path("storms_dir")        # config/paths.yaml
STORM_NAMES = ["Debi", "Mathis", "Ciaran", "Nelson"]
OUT_PNG = f"{path('root')}/Outputs/storms_comparison_2x2_cartopy.png"
apply_overrides(globals(), "storms", final=True)

storms = {n: dict(nc=f"{STORMS_DIR}/Storm_{n}.nc", csv=f"{STORMS_DIR}/Storm_{n}.csv")
          for n in STORM_NAMES}

def load_and_crop(nc_path, pad=20):
    """Load footprint, crop to valid (non-NaN) swath + padding."""
    ds = xr.open_dataset(nc_path)
    fp = ds['decontaminated_footprint'].isel(track_id=0)
    valid = fp.notnull()
    lat_idx = np.where(valid.any(axis=1))[0]
    lon_idx = np.where(valid.any(axis=0))[0]
    lat_sl = slice(max(lat_idx.min() - pad, 0), min(lat_idx.max() + pad, fp.shape[0]))
    lon_sl = slice(max(lon_idx.min() - pad, 0), min(lon_idx.max() + pad, fp.shape[1]))
    return ds, fp[lat_sl, lon_sl]

def main():
    # --- Load everything first ---
    data = {}
    for name, paths in storms.items():
        ds, fp_c = load_and_crop(paths['nc'])
        track = pd.read_csv(paths['csv'])
        data[name] = dict(ds=ds, fp=fp_c, track=track)

    vmin = 10
    vmax = max(d['fp'].max().item() for d in data.values())  # shared color scale

    # --- Compute one shared extent covering the union of all 4 storms ---
    lon_min = min(d['fp'].longitude.min().item() for d in data.values())
    lon_max = max(d['fp'].longitude.max().item() for d in data.values())
    lat_min = min(d['fp'].latitude.min().item() for d in data.values())
    lat_max = max(d['fp'].latitude.max().item() for d in data.values())
    extent_pad = 2  # degrees
    shared_extent = [lon_min - extent_pad, lon_max + extent_pad,
                      lat_min - extent_pad, lat_max + extent_pad]

    # =========================================================
    # Plot 2x2 grid with cartopy
    # =========================================================
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(15, 13),
                              subplot_kw={'projection': proj})

    for ax, (name, d) in zip(axes.flat, data.items()):
        fp_c, track, ds = d['fp'], d['track'], d['ds']

        ax.set_extent(shared_extent, crs=proj)

        # Basemap: ocean + land + coastlines
        ax.add_feature(cfeature.OCEAN, facecolor='#dceaf5', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#e8e4d8', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', zorder=2)

        mesh = ax.pcolormesh(fp_c.longitude, fp_c.latitude, fp_c, cmap='YlOrRd',
                              shading='auto', vmin=vmin, vmax=vmax,
                              transform=proj, zorder=1)

        ax.plot(track['longitude'], track['latitude'], color='blue', lw=1.5,
                marker='o', ms=6, transform=proj, zorder=3, label='Track center')

        for _, row in track.iloc[::2].iterrows():
            ax.annotate(pd.to_datetime(row['time']).strftime('%d %Hh'),
                        xy=(row['longitude'], row['latitude']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=8, color='navy', fontweight='bold',
                        transform=proj)

        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                           alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'weight': 'bold'}
        gl.ylabel_style = {'size': 12, 'weight': 'bold'}

        central_time = str(ds['footprint_central_time'].values[0])[:16]
        ax.set_title(f"Storm {name}\ncentral time: {central_time}", fontsize=12)
        ax.legend(loc='lower right', fontsize=8)

    # --- Shared colorbar ---
    fig.subplots_adjust(right=0.90, wspace=0.2, hspace=0.01)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mesh, cax=cbar_ax, label='Max 10 m wind gust (m/s), 72h window')

    #fig.suptitle('European windstorm gust footprints — 2023/24 season', fontsize=16, y=0.98)
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('saved', OUT_PNG)


if __name__ == "__main__":
    main()
