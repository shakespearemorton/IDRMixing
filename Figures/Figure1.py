#!/usr/bin/env python3
"""Figure 1: 2x4 layout - A: spatial+KDE, B: R² curves, C-D: scatter, E-H: pair analyses"""
import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde
from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap
from glob import glob

plt.rcParams.update({
    'font.size': 20, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 8, 'xtick.major.width': 2, 'ytick.major.size': 8, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

COLORS = {'dense': '#e02b35', 'mid': '#EBECED', 'dilute': '#2066a8',
          'comp1': '#a559aa', 'comp2': '#59a89c', 'compalt': '#A89759'}
ATOMS_A3_TO_mM = 1.660539e6
DATA_DIR = Path("figure1_data")

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

def load_calculated_data():
    """Load density data from results_35."""
    data = {}
    for f in Path("../Figure0/results_35").glob("chunk_density_data_*.npz"):
        name = f.stem.replace("chunk_density_data_", "")
        with np.load(f) as d:
            data[name] = {'hist_den': float(d['hist_den']), 'hist_dil': float(d['hist_dil'])}
    return data

def r2_for_cube(folder, df_ref):
    """Return (R²_dense, R²_dilute) or None."""
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
    r_den, _ = pearsonr(df_valid['cden'], df_valid['pred_den'])
    r_dil, _ = pearsonr(df_valid['cdil'], df_valid['pred_dil'])
    return r_den**2, r_dil**2

def plot_panel_A(fig, ax):
    """Panel A: Spatial map + KDE."""
    direc, chunk_size = Path('../Figure0/Q3LI60_1_44'), 35
    u = mda.Universe(str(direc / "top.pdb"), str(list(direc.glob("*.dcd"))[0]))
    u.trajectory[-1]
    
    box = u.dimensions[:3]
    n_chunks = np.maximum(np.round(box / chunk_size).astype(int), 1)
    chunk_dims = box / n_chunks
    pos = u.atoms.positions
    idx = np.clip((pos // chunk_dims).astype(int), 0, n_chunks - 1)
    
    counts_3d = np.zeros(n_chunks, dtype=np.float64)
    for i, j, k in idx: counts_3d[i, j, k] += 1
    concentration_zx = np.mean(counts_3d, axis=1).T

    data = np.load('../Figure0/results_35/chunk_density_data_Q3LI60_1_44.npz')
    x_range = data['x_range']
    kde_y = data['kde_y']
    peaks, _ = find_peaks(kde_y, height=1e-8)
    c_dilute, c_dense = x_range[peaks[0]], x_range[peaks[-1]] if len(peaks) > 1 else x_range[peaks[0]]

    cmap = LinearSegmentedColormap.from_list('custom', 
        [(0.0, COLORS['dilute']), (0.15, COLORS['mid']), (1.0, COLORS['dense'])])
    
    ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), height_ratios=[3, 1], hspace=0.0)
    
    # Spatial map
    ax_map = fig.add_subplot(gs_sub[0])
    ax_map.imshow(concentration_zx.T, origin='lower', extent=[0, box[2], 0, box[0]], 
                  aspect='equal', cmap=cmap, vmin=x_range.min(), vmax=x_range.max())
    for i in range(n_chunks[2] + 1): ax_map.axvline(i * chunk_dims[2], color='k', ls=':', lw=0.5, alpha=0.5)
    for i in range(n_chunks[0] + 1): ax_map.axhline(i * chunk_dims[0], color='k', ls=':', lw=0.5, alpha=0.5)
    ax_map.set_xlabel('Z (Å)'); ax_map.set_ylabel('X (Å)')

    # KDE
    ax_kde = fig.add_subplot(gs_sub[1])
    for i in range(len(x_range)-1):
        norm = (x_range[i] - x_range.min()) / (x_range.max() - x_range.min())
        ax_kde.fill_between(x_range[i:i+2], 0, kde_y[i:i+2], color=cmap(norm), alpha=0.6)
    ax_kde.plot(x_range, kde_y, 'k-', lw=1.5)
    ax_kde.axvline(c_dilute, color=COLORS['dilute'], ls='--', lw=1.5, alpha=0.8)
    if len(peaks) > 1: ax_kde.axvline(c_dense, color=COLORS['dense'], ls='--', lw=1.5, alpha=0.8)
    y_max = kde_y.max()
    ax_kde.text(c_dilute, y_max*1.05, f'{c_dilute:.1f}', ha='center', va='bottom', color=COLORS['dilute'], fontweight='bold')
    if len(peaks) > 1: ax_kde.text(c_dense, y_max*1.05, f'{c_dense:.1f}', ha='center', va='bottom', color=COLORS['dense'], fontweight='bold')
    ax_kde.set_xlabel('Concentration (Res./Voxel)'); ax_kde.set_ylabel('Prob. Density')
    
    save_csv({'concentration_zx_flat': concentration_zx.flatten(), 'kde_x': x_range, 'kde_y': kde_y,
              'c_dilute': c_dilute, 'c_dense': c_dense}, 'panelA_composite_data.csv')

def plot_pair_composite(fig, ax, pair_id, comp_label, colors):
    """Panels E, G: Probability densities for a pair."""
    chunk_size, box = 35, np.array([200, 200, 2000])
    chunk_vol = np.prod(box / np.maximum(np.round(box / chunk_size).astype(int), 1))
    
    active_data = np.load(f"../Figure1/active_data_final/{pair_id}_{comp_label}_active_data.npy")
    frac_s1 = active_data[:, 4]
    densities = chunk_vol * active_data[:, 0] / ATOMS_A3_TO_mM
    d_s1, d_s2 = densities * frac_s1, densities * (1 - frac_s1)
    x_range = np.linspace(0, densities.max(), 500)
    
    kdes = [gaussian_kde(d, bw_method='scott')(x_range) for d in [densities, d_s1, d_s2]]
    peaks, _ = find_peaks(kdes[0], height=6e-8)
    peaks = sorted(peaks, key=lambda p: kdes[0][p], reverse=True)
    
    ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=ax.get_subplotspec(), hspace=0.4)
    labels = ['Combined', 'IDR-1', 'IDR-2']
    fill_colors = [None, colors[0], colors[1]]
    line_colors = ['k', colors[0], colors[1]]
    
    for i, (kde_y, fill, line, label) in enumerate(zip(kdes, fill_colors, line_colors, labels)):
        ax_kde = fig.add_subplot(gs_sub[i])
        if fill: ax_kde.fill_between(x_range, 0, kde_y, color=fill, alpha=0.6)
        ax_kde.plot(x_range, kde_y, color=line, lw=1.5)
        if i == 0:
            for pk in peaks:
                if abs(x_range[pk] - 37) >= 1:
                    ax_kde.axvline(x_range[pk], color='k', ls='--', lw=1.5, alpha=0.8)
                    ax_kde.text(x_range[pk], kde_y.max()*1.05, f'{x_range[pk]:.1f}', ha='center', va='bottom', fontweight='bold')
        ax_kde.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_kde.set_ylabel(label, fontsize=15, rotation=90, labelpad=20, va='center')
        if i == 2: ax_kde.set_xlabel('Concentration (Res./Voxel)')
    
    return densities, d_s1, d_s2

def plot_pair_scatter(fig, ax, d, d1, d2, s1, s2, colors):
    """Panels F, H: Scatter with marginal KDEs."""
    maxi = np.max(d)
    ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=ax.get_subplotspec(), hspace=0.05, wspace=0.05,
                                              height_ratios=[0.5, 1, 1, 1], width_ratios=[1, 1, 1, 0.5])
    
    ax_main = fig.add_subplot(gs_sub[1:, :-1])
    ax_main.scatter(d1, d2, alpha=0.4, s=40, color='grey', edgecolors='black', lw=0.5, rasterized=True)
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xlim(0, maxi); ax_main.set_ylim(0, maxi)
    ax_main.set_yticks(ax_main.get_xticks())
    for t in ax_main.get_xticks():
        if 0 <= t <= maxi:
            ax_main.plot([t, t], [0, max(0, maxi-t)], color='gray', ls='--', alpha=0.3, lw=0.8)
            ax_main.plot([0, max(0, maxi-t)], [t, t], color='gray', ls='--', alpha=0.3, lw=0.8)
    ax_main.plot([0, maxi], [maxi, 0], 'k--', alpha=0.5, lw=2, zorder=10)
    ax_main.set_xlabel(f'{s1} (Res./Voxel)'); ax_main.set_ylabel(f'{s2} (Res./Voxel)')
    
    rng = np.linspace(0, maxi, 200)
    ax_top = fig.add_subplot(gs_sub[0, :-1], sharex=ax_main)
    kde1 = gaussian_kde(d1)(rng)
    ax_top.fill_between(rng, kde1, alpha=0.5, color=colors[0])
    ax_top.plot(rng, kde1, color=colors[0], lw=2)
    ax_top.set_xlim(0, maxi); ax_top.axis('off')
    
    ax_right = fig.add_subplot(gs_sub[1:, -1], sharey=ax_main)
    kde2 = gaussian_kde(d2)(rng)
    ax_right.fill_betweenx(rng, kde2, alpha=0.5, color=colors[1])
    ax_right.plot(kde2, rng, color=colors[1], lw=2)
    ax_right.set_ylim(0, maxi); ax_right.axis('off')

def plot_scatter_panel(ax, df, df_valid, df_grey, xcol, ycol, color, xlabel, ylabel):
    """Helper for panels C and D."""
    ax.scatter(df_grey[xcol], df_grey[ycol], color='grey', s=150, alpha=0.5, edgecolors='darkgrey', lw=1)
    ax.scatter(df_valid[xcol], df_valid[ycol], color=color, s=150, alpha=0.7, edgecolors='black', lw=1)
    lo = min(df[xcol].min(), df[ycol].min())
    hi = max(df[xcol].max(), df[ycol].max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, lw=2)
    if len(df_valid) >= 2:
        r, _ = stats.pearsonr(df_valid[xcol], df_valid[ycol])
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

def find_closest_5050_comp(pair_id):
    """Find composition closest to 50/50."""
    files = glob(f"../Figure1/active_data_final/{pair_id}_*_active_data.npy")
    comps = []
    for f in files:
        comp_str = os.path.basename(f).replace(f"{pair_id}_", "").replace("_active_data.npy", "")
        c1, c2 = map(int, comp_str.split('_'))
        comps.append((c1, c2))
    closest = min(comps, key=lambda c: abs(c[0]/(c[0]+c[1]) - 0.5))
    return f"{closest[0]}_{closest[1]}"

def create_figure1():
    """Create main Figure 1."""
    df_training = pd.read_csv('../DATASETS/df_training.csv')
    df_prep = pd.read_csv("../DATASETS/gin_prep.csv")
    
    # Collect R² data
    cube_data = []
    for folder in sorted(Path('.').glob('../Figure0/results_*')):
        m = re.fullmatch(r'results_(\d+(?:\.\d+)?)', folder.name)
        if m:
            r_pair = r2_for_cube(folder, df_training)
            if r_pair: cube_data.append((float(m.group(1)), *r_pair))
    cube_data.sort()
    cube_sizes, r2_dense, r2_dil = (np.array(x) for x in zip(*cube_data)) if cube_data else (np.array([]),)*3
    
    # Load scatter plot data
    calc_data = load_calculated_data()
    rows = []
    for name, d in calc_data.items():
        match = df_training[df_training['seq_name'] == name]
        if not match.empty:
            cden_val = float(match['cden'].iloc[0])
            cden_nan = np.isnan(cden_val)
            rows.append({'cden': 0.0 if cden_nan else cden_val, 'cdil': match['cdil'].iloc[0],
                        'hist_den': d['hist_den'], 'hist_dil': d['hist_dil'], 'cden_was_nan': cden_nan})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['cdil', 'hist_den', 'hist_dil'])
    df_valid, df_grey = df[~df['cden_was_nan']], df[df['cden_was_nan']]
    
    # Create figure
    fig = plt.figure(figsize=(30, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)
    add_label = lambda ax, lbl: ax.text(-0.1, 1.2, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel A
    ax = fig.add_subplot(gs[0, 0]); add_label(ax, 'A')
    plot_panel_A(fig, ax)
    
    # Panel B
    ax = fig.add_subplot(gs[0, 1]); add_label(ax, 'B')
    ax.plot(cube_sizes, r2_dense, 'o-', lw=3, color=COLORS['dense'], label='$c_{den}$', ms=8)
    ax.plot(cube_sizes, r2_dil, 's-', lw=3, color=COLORS['dilute'], label='$c_{dil}$', ms=8)
    ax.axvline(35, color='grey', ls='--', lw=2, label='35 Å')
    ax.legend()
    ax.set_xlabel('Voxel Side Length - S (Å)'); ax.set_ylabel('Comp. w/ Established Methods ($R^{2}$)')
    ax.set_ylim(0.75, 1.05)
    save_csv({'cube_sizes': cube_sizes, 'r2_dense': r2_dense, 'r2_dil': r2_dil}, 'panelB_r2_data.csv')
    
    # Panel C
    ax = fig.add_subplot(gs[0, 2]); add_label(ax, 'C')
    plot_scatter_panel(ax, df, df_valid, df_grey, 'cden', 'hist_den', COLORS['dense'],
                      '$c_{den}$ (Gibbs Dividing Surface) (mM)', '$c_{den}$ (Domain Decomposition) (mM)')
    save_csv({'cden': df['cden'], 'hist_den': df['hist_den'], 'is_grey': df['cden_was_nan']}, 'panelC_dense_scatter.csv')
    
    # Panel D
    ax = fig.add_subplot(gs[0, 3]); add_label(ax, 'D')
    plot_scatter_panel(ax, df, df_valid, df_grey, 'cdil', 'hist_dil', COLORS['dilute'],
                      '$c_{dil}$ (Gibbs Dividing Surface) (mM)', '$c_{dil}$ (Domain Decomposition) (mM)')
    save_csv({'cdil': df['cdil'], 'hist_dil': df['hist_dil'], 'is_grey': df['cden_was_nan']}, 'panelD_dilute_scatter.csv')
    
    # Panels E-H: Pair analyses
    pairs = [('Q96K19_200_258_Q3LI60_1_44', [COLORS['compalt'], COLORS['comp2']], 'E', 'F'),
             ('Q8WTT2_1_202_Q3LI60_1_44', [COLORS['comp1'], COLORS['comp2']], 'G', 'H')]
    
    for i, (pair_id, clrs, lbl1, lbl2) in enumerate(pairs):
        pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair_id].iloc[0]
        s1, s2 = pair_row['seq_name1'], pair_row['seq_name2']
        comp_label = find_closest_5050_comp(pair_id)
        
        ax = fig.add_subplot(gs[1, 2*i]); add_label(ax, lbl1)
        d, d1, d2 = plot_pair_composite(fig, ax, pair_id, comp_label, clrs)
        
        ax = fig.add_subplot(gs[1, 2*i+1]); add_label(ax, lbl2)
        plot_pair_scatter(fig, ax, d, d1, d2, s1, s2, clrs)
        save_csv({'densities_total': d, 'densities_s1': d1, 'densities_s2': d2}, f'panel{lbl1}{lbl2}_{pair_id}_data.csv')
    
    plt.tight_layout()
    plt.savefig('Figure1.pdf', bbox_inches='tight')
    print("Figure 1 saved as Figure1.pdf")

if __name__ == '__main__':
    create_figure1()