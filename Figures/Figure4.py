#!/usr/bin/env python3
"""Figure 4: FUS-A1 analysis, demixing predictions, and network topology"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from glob import glob
from pathlib import Path
from collections import defaultdict
from scipy.stats import gaussian_kde, spearmanr, linregress
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import cdist
from matplotlib.patches import Patch
from pycirclize import Circos
from joblib import load
from localcider.sequenceParameters import SequenceParameters
import itertools

plt.rcParams.update({
    'font.size': 20, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 8, 'xtick.major.width': 2, 'ytick.major.size': 8, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

DATA_DIR = Path("figure4_data")
ATOMS_A3_TO_mM = 1.660539e6
CHUNK_SIZE, BOX = 35, np.array([200, 200, 2000])
CHUNK_VOL = np.prod(BOX / np.maximum(np.round(BOX / CHUNK_SIZE).astype(int), 1))

COLORS = {'ratio_2': '#A6736A', 'ratio_7': '#5A708C', 'normal': '#2c3e50', 'ewsr1_fus': '#27ae60', 'ctr9': '#e67e22'}

PROTEIN_ID_MAP = {
    'CCNT1_HUMAN': 'Cyclin T1', 'DDX3X_HUMAN': 'Ddx3', 'DDX4_HUMAN': 'Ddx4', 'DYRK3_HUMAN': 'DYRK3',
    'FUS_HUMAN': 'FUS', 'G3BP1_HUMAN': 'G3BP1', 'GATA3_HUMAN': 'GATA3', 'ROA1_HUMAN': 'hnRNPA1',
    'CBX5_HUMAN': 'HP1α', 'NPHN_HUMAN': 'Nephrin', 'NPM_HUMAN': 'NPM1', 'NUP98_HUMAN': 'Nup98',
    'RBM14_HUMAN': 'RBM-14', 'SYN1_HUMAN': 'Syn-1', 'SYN2_HUMAN': 'Syn-2', 'RBP56_HUMAN': 'TAF-15',
    'TADBP_HUMAN': 'TDP-43', 'UBQL2_HUMAN': 'UBQ2', 'HSPB8_HUMAN': 'HSPB8', 'ESR1_HUMAN': 'Erα'
}

# Load residue parameters for sequence feature calculation
RESIDUES = pd.read_csv("../DATASETS/residues.csv").set_index('one')
_FEATURE_CACHE = {}  # Cache computed features

def save_csv(data_dict, filename):
    DATA_DIR.mkdir(exist_ok=True)
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 for v in data_dict.values())
    for k, v in data_dict.items():
        if not hasattr(v, '__len__') or isinstance(v, str): data_dict[k] = [v] * max_len
        elif len(v) < max_len: data_dict[k] = list(v) + [np.nan] * (max_len - len(v))
    pd.DataFrame(data_dict).to_csv(DATA_DIR / filename, index=False)

def calc_seq_features(seq, Nc=0, Cc=0, Hc=0):
    """Calculate sequence features for demixing model prediction.
    G. Tesei, A. I. Trolle, N. Jonsson, J. Betz, F. E. Knudsen, F. Pesce, K. E. Johansson, K. Lindorff-Larsen Conformational ensembles of the human intrinsically disordered proteome Nature. 2024 626:897–904 2023.05.08.539815 DOI: https://doi.org/10.1038/s41586-023-07004-5
    G. Tesei and K. Lindorff-Larsen Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range [version 2; peer review: 2 approved] Open Research Europe 2023 2(94) DOI: https://doi.org/10.12688/openreseurope.14967.2"""
    seq_str = str(seq).upper()
    cache_key = (seq_str, Nc, Cc, Hc)
    if cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]
    
    seq = [aa for aa in seq_str if aa in RESIDUES.index]
    if len(seq) == 0: return None
    N = len(seq)
    r = RESIDUES.copy()
    fasta_kappa = np.array(seq.copy())
    
    # Calculate properties that do not depend on charges
    fK = sum([seq.count(a) for a in ['K']]) / N
    fR = sum([seq.count(a) for a in ['R']]) / N
    fE = sum([seq.count(a) for a in ['E']]) / N
    fD = sum([seq.count(a) for a in ['D']]) / N
    faro = sum([seq.count(a) for a in ['W', 'Y', 'F']]) / N
    mean_lambda = np.mean(r.loc[seq, 'lambdas'])
    
    pairs = np.array(list(itertools.combinations(seq, 2)))
    pairs_indices = np.array(list(itertools.combinations(range(N), 2)))
    ij_dist = np.diff(pairs_indices, axis=1).flatten().astype(float)
    ll = r.loc[pairs[:, 0], 'lambdas'].values + r.loc[pairs[:, 1], 'lambdas'].values
    shd = np.sum(ll * np.power(np.abs(ij_dist), -1)) / N
    
    # Fix charges
    if Nc == 1:
        r.loc['X'] = r.loc[seq[0]]
        r.loc['X', 'q'] = r.loc[seq[0], 'q'] + 1.
        seq[0] = 'X'
        fasta_kappa[0] = 'K' if r.loc['X', 'q'] > 0 else 'A'
    if Cc == 1:
        r.loc['Z'] = r.loc[seq[-1]]
        r.loc['Z', 'q'] = r.loc[seq[-1], 'q'] - 1.
        seq[-1] = 'Z'
        fasta_kappa[-1] = 'D' if r.loc['Z', 'q'] < 0 else 'A'
    if Hc < 0.5:
        r.loc['H', 'q'] = 0
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'A'
    elif Hc >= 0.5:
        r.loc['H', 'q'] = 1
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'K'
    
    # Calculate properties that depend on charges
    pairs = np.array(list(itertools.combinations(seq, 2)))
    qq = r.loc[pairs[:, 0], 'q'].values * r.loc[pairs[:, 1], 'q'].values
    scd = np.sum(qq * np.sqrt(ij_dist)) / N
    
    SeqOb = SequenceParameters(''.join(fasta_kappa))
    kappa = SeqOb.get_kappa()
    fcr = r.loc[seq, 'q'].abs().mean()
    ncpr = r.loc[seq, 'q'].mean()
    
    result = {'N': N, 'fK': fK, 'fR': fR, 'fE': fE, 'fD': fD, 'faro': faro, 
              'scd': scd, 'shd': shd, 'kappa': kappa, 'fcr': fcr, 'mean_lambda': mean_lambda, 'ncpr': ncpr}
    _FEATURE_CACHE[cache_key] = result
    return result

def predict_demixing(seq1, seq2, pipeline, features):
    """Predict demixing using the Figure3 model."""
    f1, f2 = calc_seq_features(seq1), calc_seq_features(seq2)
    if f1 is None or f2 is None: return None
    
    # Build feature vector matching model expectations
    X = pd.DataFrame([{f'{s}_mean': (f1[s] + f2[s]) / 2 for s in ['mean_lambda', 'scd', 'N', 'faro', 'ncpr', 'kappa', 'shd']} |
                      {f'{s}_abs_diff': abs(f1[s] - f2[s]) for s in ['mean_lambda', 'scd', 'N', 'faro', 'ncpr', 'kappa', 'shd']}])
    
    return pipeline.predict_proba(X[features])[:, 1][0]

def load_model():
    """Load the demixing model from Figure3."""
    model_path = Path("../Figures/figure3_data/demixing_model_pipeline.joblib")
    if not model_path.exists():
        model_path = Path("figure3_data/demixing_model_pipeline.joblib")
    pipeline = load(model_path)
    
    # Get feature order from model specification
    spec_path = model_path.parent / "model_specification.json"
    if spec_path.exists():
        spec = pd.read_json(spec_path)
        features = spec.iloc[0]['features_in_order']
    else:
        features = list(pipeline.named_steps['model'].feature_names_in_)
    return pipeline, features

def get_volume_fractions(active_data):
    frac_s1 = active_data[:, 4]
    d = CHUNK_VOL * active_data[:, 0] / ATOMS_A3_TO_mM
    return d, d * frac_s1, d * (1 - frac_s1)

def get_mixed_phases(phi1, phi2, densities):
    """Determine phase boundaries from density distribution."""
    kde = gaussian_kde(densities)
    x_range = np.linspace(densities.min() * 0.95, densities.max() * 1.05, 4096)
    kde_y = kde(x_range)
    peaks, _ = find_peaks(kde_y)
    if len(peaks) == 0: return None, None
    
    peak_xs = x_range[peaks]
    c_dilute = peak_xs[np.argmin(peak_xs)]
    widths = peak_widths(kde_y, peaks)[0]
    dilute_thresh = c_dilute + widths[np.argmin(peak_xs)] * 0.2
    
    # Find peaks in individual component distributions
    def find_comp_peaks(phi):
        kde_p = gaussian_kde(phi)
        x = np.linspace(phi.min(), phi.max(), 4096)
        pks, _ = find_peaks(kde_p(x))
        return x[pks] if len(pks) > 0 else [phi.mean()]
    
    phi1_peaks, phi2_peaks = find_comp_peaks(phi1), find_comp_peaks(phi2)
    dilute = [phi1_peaks[0], phi2_peaks[0]]
    dense = [phi1_peaks[-1] if len(phi1_peaks) > 1 else phi1_peaks[0],
             phi2_peaks[-1] if len(phi2_peaks) > 1 else phi2_peaks[0]]
    return dilute, dense

def get_pair_data(pair_id):
    """Load simulation data for a protein pair."""
    df_prep = pd.read_csv("../DATASETS/known_gin_prep.csv")
    pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair_id]
    if pair_row.empty: return None, None
    
    s1, s2 = pair_row.iloc[0]['seq_name1'], pair_row.iloc[0]['seq_name2']
    files = glob(f"../../mix_affinity/run_known/active_data_final/{pair_id}_*_active_data.npy")
    
    compositions = []
    for f in files:
        comp_str = os.path.basename(f).replace(f"{pair_id}_", "").replace("_active_data.npy", "")
        c1, c2 = map(int, comp_str.split('_'))
        if c1 == 0 or c2 == 0: continue
        
        d, phi1, phi2 = get_volume_fractions(np.load(f))
        dilute, dense = get_mixed_phases(phi1, phi2, d)
        compositions.append({'comp1': c1, 'comp2': c2, 'phi1': phi1, 'phi2': phi2,
                            'dilute_phases': dilute, 'dense_phases': dense})
    
    return sorted(compositions, key=lambda x: x['comp1'])[:5], (s1, s2)

def get_experimental_data():
    """FUS-A1 experimental data from Farag et al."""
    df_dil = pd.DataFrame([(100, 1.780, 0.194, 0.000, 0.000), (75, 0.385, 0.011, 0.036, 0.000),
                           (50, 0.143, 0.042, 0.117, 0.000), (25, 0.007, 0.000, 0.319, 0.000),
                           (0, 0.000, 0.000, 0.763, 0.000)], columns=['Ratio', 'FUS', 'FUS_err', 'A1', 'A1_err'])
    df_den = pd.DataFrame([(75, 194.059, 66.806, 84.549, 27.480), (50, 150.107, 29.506, 134.733, 23.796),
                           (25, 59.757, 22.436, 147.483, 44.416)], columns=['Ratio', 'FUS', 'FUS_err', 'A1', 'A1_err'])
    return df_dil, df_den

def plot_fus_a1_panels(fig, gs_main):
    """Panels A-B: FUS-A1 simulation vs experiment."""
    comps_1, _ = get_pair_data('FUS_A1')
    comps_2, _ = get_pair_data('FUS_A1p12D')
    if comps_1 is None: print("FUS-A1 data not found"); return
    
    gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main, wspace=0.3)
    ax_sim, ax_exp = fig.add_subplot(gs_sub[0, 0]), fig.add_subplot(gs_sub[0, 1])
    axins_sim, axins_exp = ax_sim.inset_axes([0.55, 0.55, 0.4, 0.4]), ax_exp.inset_axes([0.55, 0.55, 0.4, 0.4])
    
    for ax, lbl in [(ax_sim, 'A'), (ax_exp, 'B')]:
        ax.text(-0.1, 1.1, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Plot simulations
    all_phi = []
    tot_phi = []
    for comps, color in [(comps_1, COLORS['ratio_2']), (comps_2, COLORS['ratio_7'])]:
        if comps is None: continue
        for c in comps:
            phi1, phi2, dil, den = c['phi1'], c['phi2'], c['dilute_phases'], c['dense_phases']
            all_phi.extend([phi1, phi2])
            tot_phi.extend(phi1+phi2)
            ax_sim.scatter(phi1, phi2, alpha=0.2, s=40, color=color, rasterized=True)
            if dil and den:
                ax_sim.plot([dil[0], den[0]], [dil[1], den[1]], color=color, lw=2, alpha=0.8)
                ax_sim.scatter(den[0], den[1], color=color, s=80, edgecolors='black', lw=1.5, zorder=5)
                axins_sim.scatter(dil[0], dil[1], color=color, s=80, edgecolors='black', lw=1.5, zorder=5)
                axins_sim.plot([dil[0], den[0]], [dil[1], den[1]], color=color, lw=2, alpha=0.8, zorder=4)
    
    maxi = max(tot_phi) * 1.1
    maxi_in = 1.5
    
    # Plot experimental
    df_dil, df_den = get_experimental_data()
    color = COLORS['ratio_2']
    for _, r in df_dil.iterrows():
        for ax in [ax_exp, axins_exp]:
            ax.errorbar(r['FUS'], r['A1'], xerr=r['FUS_err'], yerr=r['A1_err'], fmt='o', color=color,
                       ms=8, mec='black', mew=1.5, capsize=3, zorder=5)
    for _, r in df_den.iterrows():
        ax_exp.errorbar(r['FUS'], r['A1'], xerr=r['FUS_err'], yerr=r['A1_err'], fmt='o', color=color,
                       ms=8, mec='black', mew=1.5, capsize=3, zorder=5)
    for ratio in [75, 50, 25]:
        d, D = df_dil[df_dil['Ratio'] == ratio].iloc[0], df_den[df_den['Ratio'] == ratio].iloc[0]
        for ax in [ax_exp, axins_exp]:
            ax.plot([d['FUS'], D['FUS']], [d['A1'], D['A1']], color=color, ls='--', lw=2, alpha=0.8)
    
    # Format
    for ax, ins, lim, lim_in, xl, yl, ttl in [
        (ax_sim, axins_sim, maxi, maxi_in, 'FUS (Res./Voxel)', 'A1 (Res./Voxel)', 'Simulated'),
        (ax_exp, axins_exp, 350, 2, 'FUS (mg/mL)', 'A1 (mg/mL)', 'Experimental [Farag et al.]')]:
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.plot([0, lim], [lim, 0], color='gray', ls='--', alpha=0.5, lw=2)
        ins.set_xlim(0, lim_in); ins.set_ylim(0, lim_in); ins.grid(True, alpha=0.3)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontweight='bold')
    
    ax_sim.legend(handles=[Patch(fc=COLORS['ratio_2'], ec='black', label='A1$_{WT}$'),
                           Patch(fc=COLORS['ratio_7'], ec='black', label='A1$_{p12D}$')], loc='center right')
    def add_gridlines(ax, limit):
        x_ticks = [tick for tick in ax.get_xticks() if 0 <= tick <= limit]
        for x in x_ticks:
            ax.plot([x, x], [0, max(0, limit - x)], color='gray', 
                    linestyle='--', alpha=0.3, linewidth=0.8)
        for y in x_ticks:
            ax.plot([0, max(0, limit - y)], [y, y], color='gray', 
                    linestyle='--', alpha=0.3, linewidth=0.8)
    add_gridlines(ax_sim, maxi)
    add_gridlines(ax_exp, 350)
    add_gridlines(axins_sim, maxi_in)
    add_gridlines(axins_exp, 2)

def plot_demixing_difference(fig, ax):
    """Panel C: ΔDemixing vs MED1 pellet ratio."""
    ax.text(-0.1, 1.1, 'C', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    df_demix = pd.read_csv("../DATASETS/sabari_demixing.csv")
    df_pellet = pd.read_csv("../DATASETS/sabari_all_pellet.csv")
    
    # Calculate ΔDemixing for each protein
    data = []
    for prot in df_demix['protein1'].unique():
        if prot in ['MED1', 'NPM1']: continue
        med1 = df_demix[((df_demix['protein1'] == prot) & (df_demix['protein2'] == 'MED1')) |
                        ((df_demix['protein1'] == 'MED1') & (df_demix['protein2'] == prot))]['demixing_composition']
        npm1 = df_demix[((df_demix['protein1'] == prot) & (df_demix['protein2'] == 'NPM1')) |
                        ((df_demix['protein1'] == 'NPM1') & (df_demix['protein2'] == prot))]['demixing_composition']
        if len(med1) > 0 and len(npm1) > 0:
            data.append({'protein': prot, 'ddemix': npm1.values[0] - med1.values[0]})
    
    df_pellet['protein'] = df_pellet.apply(lambda x: str(x['Gene name']).strip() if pd.notna(x['Gene name']) else str(x['Uniprot ID']), axis=1)
    df_plot = pd.merge(pd.DataFrame(data), df_pellet[['protein', 'Log2(average_P/S)']].rename(columns={'Log2(average_P/S)': 'pellet_ratio'}))
    
    if df_plot.empty: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return
    
    outliers = {'EWSR1': ('ewsr1_fus', 's'), 'FUS': ('ewsr1_fus', 's'), 'CTR9': ('ctr9', '^')}
    normal = df_plot[~df_plot['protein'].isin(outliers)]
    
    ax.scatter(normal['ddemix'], normal['pellet_ratio'], s=200, c=COLORS['normal'], edgecolors='k', alpha=0.7)
    for _, r in df_plot[df_plot['protein'].isin(outliers)].iterrows():
        c, m = outliers[r['protein']]
        ax.scatter(r['ddemix'], r['pellet_ratio'], s=250, c=COLORS[c], edgecolors='k', marker=m, zorder=3)
    
    if len(normal) > 1:
        res = linregress(normal['ddemix'], normal['pellet_ratio'])
        x_rng = np.array([df_plot['ddemix'].min(), df_plot['ddemix'].max()])
        ax.plot(x_rng, res.intercept + res.slope * x_rng, 'k--', alpha=0.5)
        rho, _ = spearmanr(normal['ddemix'], normal['pellet_ratio'])
        ax.text(0.05, 0.95, f'ρ = {rho:.2f}', transform=ax.transAxes, va='top', bbox=dict(fc='white', alpha=0.8))
    
    ax.set_xlabel('ΔD (NPM1 - MED1)'); ax.set_ylabel('MED1 Pellet / Supernatant'); ax.set_ylim(-5, 15)
    for _, r in df_plot.iterrows():
        ax.annotate(r['protein'], (r['ddemix'], r['pellet_ratio']), xytext=(5, 5), textcoords='offset points', fontsize=12)
    save_csv(df_plot.to_dict('list'), 'panelC_demixing_difference.csv')

def plot_prediction_accuracy(fig, ax, pipeline, features):
    """Panel D: Prediction accuracy confusion matrix."""
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    results = []
    for name, path in [('sabari', '../DATASETS/sabari_demixing.csv'), ('exp', '../DATASETS/exp_demixing.csv'),
                       ('known', '../DATASETS/known_demixing.csv')]:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            if pd.isna(r.get('seq1')) or pd.isna(r.get('seq2')) or pd.isna(r.get('demixing_composition')): continue
            pred = predict_demixing(r['seq1'], r['seq2'], pipeline, features)
            if pred is not None:
                results.append({'actual': r['demixing_composition'], 'predicted': pred, 'dataset': name})
    
    if not results: ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes); return
    
    df = pd.DataFrame(results)
    df['actual_class'], df['pred_class'] = (df['actual'] > 0.5).astype(int), (df['predicted'] > 0.5).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['actual_class'], df['pred_class'])
    ax.imshow(cm, cmap='Blues', alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Mix', 'Pred Demix'], fontsize=14)
    ax.set_yticklabels(['Actual Mix', 'Actual Demix'], fontsize=14, rotation=90)
    acc = (df['actual_class'] == df['pred_class']).mean()
    ax.text(0.5, 0.05, f'Accuracy: {acc:.1%} (n={len(df)})', transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    save_csv(df.to_dict('list'), 'panelD_predictions.csv')

def plot_network(fig, ax):
    """Panel E: Network topology of mixing interactions."""
    ax.text(-0.1, 1.05, 'E', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    path = "../DATASETS/exp_demixing.csv"
    if not os.path.exists(path): ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes); return
    
    df = pd.read_csv(path)
    df['protein1'], df['protein2'] = df['protein1'].replace(PROTEIN_ID_MAP), df['protein2'].replace(PROTEIN_ID_MAP)
    all_proteins = sorted(set(df['protein1']) | set(df['protein2']))
    
    adj_df = pd.DataFrame(0.0, index=all_proteins, columns=all_proteins)
    degree, edge_strength = defaultdict(int), {}
    
    for _, r in df.iterrows():
        if r.get('dGij', 0) < -3.5 and r.get('demixing_composition', 1) < 0.2:
            u, v = tuple(sorted([r['protein1'], r['protein2']]))
            adj_df.at[u, v] = 1 - 2 * r['demixing_composition']
            edge_strength[(u, v)] = abs(r['dGij'])
            degree[r['protein1']] += 1; degree[r['protein2']] += 1
    
    sorted_proteins = sorted(all_proteins, key=lambda x: degree.get(x, 0))
    adj_df = adj_df.reindex(index=sorted_proteins, columns=sorted_proteins)
    np.fill_diagonal(adj_df.values, 1)
    
    s_min, s_max = (min(edge_strength.values()), max(edge_strength.values())) if edge_strength else (0, 1)
    def link_kws(f, t):
        if f == t: return dict(ec='none', lw=0, alpha=0)
        raw = edge_strength.get(tuple(sorted([f, t])), 0)
        alpha = 0.1 + 0.7 * ((raw - s_min) / (s_max - s_min) if s_max > s_min else 0.5)
        return dict(ec='k', lw=0.2, alpha=alpha)
    
    active = {p for p in all_proteins if degree.get(p, 0) > 0}
    degs = [degree[p] for p in active]
    norm = mcolors.Normalize(vmin=min(degs), vmax=max(degs)) if degs else mcolors.Normalize(0, 1)
    cmap = plt.get_cmap('magma')
    colors = {n: mcolors.to_hex(cmap(norm(degree[n]))) for n in active}
    
    pos = ax.get_position(); ax.remove()
    ax_polar = fig.add_axes(pos, projection='polar')
    circos = Circos.chord_diagram(adj_df, space=2, order=sorted_proteins, cmap=colors,
                                   label_kws=dict(r=110, size=16), link_kws_handler=link_kws)
    circos.plotfig(ax=ax_polar)

def create_figure4():
    """Create Figure 4."""
    DATA_DIR.mkdir(exist_ok=True)
    pipeline, features = load_model()
    
    fig = plt.figure(figsize=(30, 21))
    gs = gridspec.GridSpec(3, 7, figure=fig, hspace=0.1, wspace=0.3)
    
    plot_fus_a1_panels(fig, gs[0:2, 0:4])
    plot_demixing_difference(fig, fig.add_subplot(gs[2, 0:2]))
    plot_prediction_accuracy(fig, fig.add_subplot(gs[2, 2:4]), pipeline, features)
    plot_network(fig, fig.add_subplot(gs[0:3, 4:7]))
    
    plt.tight_layout()
    plt.savefig('Figure4.pdf', bbox_inches='tight')
    print(f"Figure 4 saved as Figure4.pdf, data in {DATA_DIR}/")

if __name__ == '__main__':
    create_figure4()