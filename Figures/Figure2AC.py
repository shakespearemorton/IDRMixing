#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

plt.rcParams.update({
    'font.size': 16, 'axes.linewidth': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 6, 'xtick.major.width': 1.5, 'ytick.major.size': 6, 'ytick.major.width': 1.5,
    'legend.frameon': False, 'figure.dpi': 300, 'figure.facecolor': 'white', 'axes.facecolor': 'white'
})

COLORS = ['#2c0703', '#890620', '#b6465f', '#da9f93', '#ebd4cb',
          '#090F1A', '#2A3A59', '#5A708C', '#D9D4BF', '#E5E3D9']
ATOMS_A3_TO_mM = 1.660539e6
CHUNK_SIZE = 35
BOX = np.array([200, 200, 2000])
DATA_DIR = Path("figure2ac_data")

def save_csv(data_dict, filename):
    """Save data to CSV, padding arrays to equal length."""
    DATA_DIR.mkdir(exist_ok=True)
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 for v in data_dict.values())
    for k, v in data_dict.items():
        if not hasattr(v, '__len__') or isinstance(v, str):
            data_dict[k] = [v] * max_len
        elif len(v) < max_len:
            data_dict[k] = list(v) + [np.nan] * (max_len - len(v))
    pd.DataFrame(data_dict).to_csv(DATA_DIR / filename, index=False)

def get_volume_fractions(active_data):
    """Calculate volume fractions from active data."""
    chunk_vol = np.prod(BOX / np.maximum(np.round(BOX / CHUNK_SIZE).astype(int), 1))
    frac_s1 = active_data[:, 4]
    d = chunk_vol * active_data[:, 0] / ATOMS_A3_TO_mM
    return d, d * frac_s1, d * (1 - frac_s1)

def analyze_single_pair(pair_id, ax, color_offset=0):
    """Analyze a single protein pair showing all compositions on one plot."""
    df_samples = pd.read_csv("../DATASETS/demixing.csv")
    df_prep = pd.read_csv("../DATASETS/gin_prep.csv")
    
    pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair_id]
    if pair_row.empty:
        print(f"Pair {pair_id} not found"); return
    
    s1, s2 = pair_row.iloc[0]['seq_name1'], pair_row.iloc[0]['seq_name2']
    
    # Load all compositions
    compositions = []
    for f in glob(f"../Figure1/active_data_final/{pair_id}_*_active_data.npy"):
        comp_str = os.path.basename(f).replace(f"{pair_id}_", "").replace("_active_data.npy", "")
        c1, c2 = map(int, comp_str.split('_'))
        if c1 == 0 or c2 == 0: continue
        
        demix_row = df_samples[(df_samples['protein1'] == s1) & (df_samples['protein2'] == s2) & (df_samples['composition1'] == c1)]
        demix = demix_row.iloc[0]['demixing_composition'] if not demix_row.empty else np.nan
        
        d, phi1, phi2 = get_volume_fractions(np.load(f))
        compositions.append({'comp1': c1, 'comp2': c2, 'phi1': phi1, 'phi2': phi2, 'demixing': demix})
    
    compositions = sorted(compositions, key=lambda x: x['comp1'])[:5]
    if not compositions:
        print(f"No valid compositions for {pair_id}"); return
    
    # Save raw data
    for comp in compositions:
        save_csv({'phi1': comp['phi1'], 'phi2': comp['phi2'], 'demixing': comp['demixing']},
                 f"{pair_id}_{comp['comp1']}_{comp['comp2']}_data.csv")
    
    # Plot
    all_phi = np.concatenate([np.concatenate([c['phi1'], c['phi2']]) for c in compositions])
    maxi = all_phi.max() * 1.1
    
    for idx, comp in enumerate(compositions):
        label = f"{comp['comp1']}:{comp['comp2']} (D = {comp['demixing']:.3f})"
        ax.scatter(comp['phi1'], comp['phi2'], alpha=0.7, s=40, color=COLORS[idx + color_offset],
                   edgecolors='black', linewidth=0.2, rasterized=True, label=label)
    
    ax.set_xlim(0, maxi); ax.set_ylim(0, maxi)
    ax.set_aspect('equal', adjustable='box')
    ax.yaxis.set_major_locator(ax.xaxis.get_major_locator())
    ax.plot([0, maxi], [maxi, 0], color='gray', ls='--', alpha=0.5, lw=2)
    
    for t in ax.get_xticks():
        if 0 <= t <= maxi:
            ax.plot([t, t], [0, max(0, maxi - t)], color='gray', ls='--', alpha=0.3, lw=0.8)
            ax.plot([0, max(0, maxi - t)], [t, t], color='gray', ls='--', alpha=0.3, lw=0.8)
    
    ax.set_xlabel(f'{s1} (Res. / Voxel)'); ax.set_ylabel(f'{s2} (Res. / Voxel)')
    ax.legend(loc='best', fontsize=15)
    print(f"Analyzed {len(compositions)} compositions for {pair_id}")

if __name__ == "__main__":
    pair_ids = ['O60229_1_34_Q96RR1_1_152', 'Q3LI60_1_44_Q92581_1_59']
    color_offsets = [5, 0]
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 16))
    for pair_id, ax, offset in zip(pair_ids, axes, color_offsets):
        analyze_single_pair(pair_id, ax, color_offset=offset)
    
    plt.savefig('Figure2AC.pdf', facecolor='white', dpi=300)
    print(f"\nFigure saved as Figure2AC.pdf, data saved in {DATA_DIR}/")