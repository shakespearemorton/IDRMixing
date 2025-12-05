#!/usr/bin/env python3
"""Extended Data Figure 1: R² curves, scatter plots, and violin comparison"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

plt.rcParams.update({
    'font.size': 20, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 8, 'xtick.major.width': 2, 'ytick.major.size': 8, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

COLORS = {'dense': '#e02b35', 'dilute': '#2066a8', 'dG': '#a559aa', 'ds1': '#ffffff', 'ds2': '#969696'}

def load_calculated_data():
    data = {}
    for f in Path("../Figure0/results_35").glob("chunk_density_data_*.npz"):
        name = f.stem.replace("chunk_density_data_", "")
        with np.load(f) as d:
            data[name] = {'hist_den': float(d['hist_den']), 'hist_dil': float(d['hist_dil'])}
    return data

def r2_for_cube(folder, df_ref):
    rows = []
    for f in folder.glob('chunk_density_data_*.npz'):
        seq = f.stem.replace('chunk_density_data_', '')
        row = df_ref[df_ref['seq_name'] == seq]
        if row.empty: continue
        with np.load(f) as d:
            if {'hist_den', 'hist_dil'} <= set(d.files):
                rows.append({'cden': float(row['cden'].iloc[0]), 'cdil': float(row['cdil'].iloc[0]),
                            'pred_den': float(d['hist_den']), 'pred_dil': float(d['hist_dil'])})
    if len(rows) < 2: return None
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    df_valid = df[df['cden'] != 0]
    if len(df_valid) < 2: return None
    return pearsonr(df_valid['cden'], df_valid['pred_den'])[0]**2, pearsonr(df_valid['cdil'], df_valid['pred_dil'])[0]**2

def plot_scatter(ax, x, y, x_grey, y_grey, color, xlabel, ylabel, xlim=None):
    """Helper for scatter plots with R² annotation."""
    ax.scatter(x_grey, y_grey, color='grey', s=150, alpha=0.5, edgecolors='darkgrey', lw=1)
    ax.scatter(x, y, color=color, s=150, alpha=0.7, edgecolors='black', lw=1)
    if xlim:
        lo, hi = xlim
    else:
        all_vals = pd.concat([x, y, x_grey, y_grey])
        lo, hi = all_vals.min(), all_vals.max()
        pad = (hi - lo) * 0.05
        lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=2)
    if len(x) >= 2:
        r, _ = pearsonr(x, y)
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

def create_extended_figure1():
    df_training = pd.read_csv('../DATASETS/df_training.csv')
    gin_samples = pd.read_csv('../DATASETS/gin_samples.csv')
    
    # Collect R² data
    cube_data = []
    for folder in sorted(Path('.').glob('../Figure0/results_*')):
        m = re.fullmatch(r'results_(\d+(?:\.\d+)?)', folder.name)
        if m:
            r_pair = r2_for_cube(folder, df_training)
            if r_pair: cube_data.append((float(m.group(1)), *r_pair))
    cube_data.sort()
    cube_sizes, r2_dense, r2_dil = (np.array(x) for x in zip(*cube_data)) if cube_data else (np.array([]),)*3
    
    # Load scatter data
    calc_data = load_calculated_data()
    rows = []
    for name, d in calc_data.items():
        match = df_training[df_training['seq_name'] == name]
        if not match.empty:
            cden_val = float(match['cden'].iloc[0])
            rows.append({'cden': 0.0 if np.isnan(cden_val) else cden_val, 'cdil': match['cdil'].iloc[0],
                        'dG': match['dG'].iloc[0], 'hist_den': d['hist_den'], 'hist_dil': d['hist_dil'],
                        'cden_nan': np.isnan(cden_val)})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['cdil', 'hist_den', 'hist_dil'])
    df['pred_dG'] = np.log(df['hist_dil'] / df['hist_den'])
    df_valid, df_grey = df[~df['cden_nan']], df[df['cden_nan']]
    
    df_dG = df[['dG', 'pred_dG', 'cden_nan']].replace([np.inf, -np.inf], np.nan).dropna()
    df_dG_valid, df_dG_grey = df_dG[~df_dG['cden_nan']], df_dG[df_dG['cden_nan']]
    
    # Create figure
    fig = plt.figure(figsize=(30, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4, height_ratios=[1, 1.2])
    add_label = lambda ax, lbl: ax.text(-0.1, 1.2, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel A: R² vs cube size
    ax = fig.add_subplot(gs[0, 0]); add_label(ax, 'A')
    if cube_sizes.size > 0:
        ax.plot(cube_sizes, r2_dense, 'o-', lw=3, color=COLORS['dense'], label='$c_{den}$', ms=8)
        ax.plot(cube_sizes, r2_dil, 's-', lw=3, color=COLORS['dilute'], label='$c_{dil}$', ms=8)
        ax.axvline(35, color='grey', ls='--', lw=2, label='35 Å')
        ax.legend()
    ax.set_xlabel('Voxel Side Length - S (Å)'); ax.set_ylabel('Comp. w/ Established Methods ($R^{2}$)')
    ax.set_ylim(0.75, 1.05)
    
    # Panel B: Dilute scatter
    ax = fig.add_subplot(gs[0, 1]); add_label(ax, 'B')
    plot_scatter(ax, df_valid['cdil'], df_valid['hist_dil'], df_grey['cdil'], df_grey['hist_dil'],
                 COLORS['dilute'], '$c_{dil}$ (Gibbs Dividing Surface) (mM)', '$c_{dil}$ (Domain Decomposition) (mM)')
    
    # Panel C: Dilute scatter zoomed
    ax = fig.add_subplot(gs[0, 2]); add_label(ax, 'C')
    df_z = df[(df['cdil'] <= 1) & (df['hist_dil'] <= 1)]
    df_zv, df_zg = df_z[~df_z['cden_nan']], df_z[df_z['cden_nan']]
    plot_scatter(ax, df_zv['cdil'], df_zv['hist_dil'], df_zg['cdil'], df_zg['hist_dil'],
                 COLORS['dilute'], '$c_{dil}$ (Gibbs Dividing Surface) (mM)', '$c_{dil}$ (Domain Decomposition) (mM)', xlim=(0, 1))
    
    # Panel D: dG scatter
    ax = fig.add_subplot(gs[0, 3]); add_label(ax, 'D')
    plot_scatter(ax, df_dG_valid['dG'], df_dG_valid['pred_dG'], df_dG_grey['dG'], df_dG_grey['pred_dG'],
                 COLORS['dG'], r'$\Delta G$ (Gibbs Dividing Surface)(k$_{B}$T)', r'$\Delta G$ (Domain Decomposition)(k$_{B}$T)', xlim=(0, -10.2))
    
    # Panel E: Violin plots
    ax = fig.add_subplot(gs[1, :]); 
    ax.set_axis_off()
    gs_violin = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=ax.get_subplotspec(), wspace=0.5)
    
    # Add label E aligned with panel A (at ~1/4 width since it spans 4 columns)
    ax.text(-0.025, 1.2, 'E', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    params = ['N', 'mean_lambda', 'dG', 'faro', 'fcr', 'ncpr']
    ylabels = {'N': 'N', 'mean_lambda': r'$\langle\lambda\rangle$', 'dG': r'$\Delta G$',
               'faro': r'$f_{\mathrm{aro}}$', 'fcr': 'FCR', 'ncpr': 'NCPR'}
    titles = {'N': 'Sequence Length', 'mean_lambda': 'Mean Residue\nHydrophobicity', 'dG': 'Transfer Free Energy',
              'faro': 'Fraction Aromatic\nResidues', 'fcr': 'Fraction Charged\nResidues', 'ncpr': 'Net Charge Per\nResidue'}
    palette = {'Our Dataset': COLORS['ds1'], 'von Bülow Dataset': COLORS['ds2']}
    
    for idx, param in enumerate(params):
        ax_v = fig.add_subplot(gs_violin[0, idx])
        data = pd.concat([pd.DataFrame({'value': gin_samples[param].values, 'source': 'Our Dataset'}),
                          pd.DataFrame({'value': df_training[param].values, 'source': 'von Bülow Dataset'})], ignore_index=True)
        data = data.dropna(subset=['value'])
        sns.violinplot(x='source', y='value', data=data, ax=ax_v, palette=palette, inner='quartile',
                       hue='source', split=True, legend=False, density_norm='width')
        sns.stripplot(x='source', y='value', data=data, ax=ax_v, color='black', size=8, alpha=0.5, jitter=True)
        ax_v.set_ylabel(ylabels[param], fontsize=18); ax_v.set_xlabel(titles[param], fontsize=18); ax_v.set_xticks([])
        ax_v.spines['top'].set_visible(False); ax_v.spines['right'].set_visible(False)
    
    fig.legend(handles=[mpatches.Patch(fc=c, ec='black', lw=1.5, label=l) for l, c in palette.items()],
               loc='upper center', bbox_to_anchor=(0.5, 0.52), ncol=2, fontsize=18)
    
    plt.tight_layout()
    plt.savefig('Supplementary_Figure1.pdf', bbox_inches='tight')
    print("Supplementary Figure 1 saved")

if __name__ == '__main__':
    create_extended_figure1()