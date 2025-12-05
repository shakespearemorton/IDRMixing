#!/usr/bin/env python3
"""Figure 2 Panels G-H: G: Box plot of demixing index, H: Bar graph by composition"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 20, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 8, 'xtick.major.width': 2, 'ytick.major.size': 8, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

BAR_COLORS = ['#cad2c5', '#84a98c', '#52796f', '#354f52', '#2f3e46']
BIN_ORDER = ['5-15', '15-35', '50', '65-85', '85-95']
DATA_DIR = Path("figure2gh_data")

def assign_bin(comp):
    """Assign composition to bins."""
    if 85 <= comp <= 100: return '85-95'
    if 65 <= comp < 85: return '65-85'
    if 35 <= comp < 65: return '50'
    if 15 <= comp < 35: return '15-35'
    if 0 <= comp < 15: return '5-15'
    return None

def load_data():
    """Load and merge datasets."""
    df_dem = pd.read_csv("../DATASETS/demixing.csv")
    df_tie = pd.read_csv("../DATASETS/tie_lines.csv")
    df_prot = pd.read_csv("../DATASETS/gin_samples.csv")
    
    df = pd.merge(df_dem, df_tie[['protein1', 'protein2', 'composition1', 'composition2', 'phase_behavior']],
                  on=['protein1', 'protein2', 'composition1', 'composition2'], how='left')
    
    cols = ['seq_name', 'faro', 'ah_ij', 'N', 'shd', 'mean_lambda', 'SPR_svr']
    for p, sfx in [('protein1', '_p1'), ('protein2', '_p2')]:
        df = pd.merge(df, df_prot[cols], left_on=p, right_on='seq_name', how='left', suffixes=('', '_temp'))
        df = df.drop(columns=['seq_name']).rename(columns={c: c + sfx for c in cols[1:]})
    
    return df[(df['dGij'] < -3) & (df['composition1'] != 100) & (df['composition2'] != 100)].copy()

def create_panels_GH():
    """Create Panels G and H."""
    DATA_DIR.mkdir(exist_ok=True)
    df = load_data()
    demix_vals = df['demixing_composition'].dropna()
    
    # Panel H data
    df['composition_bin'] = df['composition1'].apply(assign_bin)
    df_h = df[df['composition_bin'].notna()]
    comp_stats = df_h.groupby('composition_bin')['demixing_composition'].agg(['mean', 'std', 'count']).reindex(BIN_ORDER)
    comp_stats['sem'] = comp_stats['std'] / np.sqrt(comp_stats['count'])
    
    # Save raw data
    demix_vals.to_csv(DATA_DIR / "panel_G_demixing_values.csv", index=False)
    comp_stats.to_csv(DATA_DIR / "panel_H_composition_stats.csv")
    df_h[['composition_bin', 'demixing_composition']].to_csv(DATA_DIR / "panel_H_raw_data.csv", index=False)
    
    # Create figure
    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(16, 3))
    add_label = lambda ax, lbl: ax.text(-0.1, 1.12, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel G: Box plot
    add_label(ax_g, 'G')
    ax_g.boxplot([demix_vals], vert=False, widths=0.5, patch_artist=True,
                 boxprops=dict(facecolor='lightgrey', alpha=0.8, linewidth=2),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2))
    ax_g.set_xlabel('Demixing Index', fontsize=18)
    ax_g.set_yticks([])
    ax_g.set_xlim(-0.05, 1.05)
    ax_g.grid(True, alpha=0.3, axis='x')
    stats_text = f'n = {len(demix_vals)}\nMean = {demix_vals.mean():.3f}\nMedian = {demix_vals.median():.3f}\nStd = {demix_vals.std():.3f}'
    ax_g.text(0.95, 0.95, stats_text, transform=ax_g.transAxes, fontsize=14, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Panel H: Bar graph
    add_label(ax_h, 'H')
    x_pos = np.arange(len(BIN_ORDER))
    for i, (pos, color) in enumerate(zip(x_pos, BAR_COLORS)):
        ax_h.bar(pos, comp_stats['mean'].iloc[i], yerr=comp_stats['sem'].iloc[i],
                 capsize=8, alpha=0.8, color=color, edgecolor='black', linewidth=2)
        ax_h.text(pos, comp_stats['mean'].iloc[i] + comp_stats['sem'].iloc[i] + 0.003,
                  f'n={int(comp_stats["count"].iloc[i])}', ha='center', va='bottom', fontsize=14)
    
    ax_h.set_xlabel('IDR-1 Composition Range (%)', fontsize=18)
    ax_h.set_ylabel('Demixing Index', fontsize=18)
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels(BIN_ORDER)
    ax_h.set_xlim(-0.6, len(BIN_ORDER) - 0.4)
    ax_h.set_ylim(0, 0.35)
    ax_h.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Figure2GH.pdf', bbox_inches='tight')
    print(f"Figure saved as Figure2GH.pdf, data saved in {DATA_DIR}/")

if __name__ == '__main__':
    create_panels_GH()