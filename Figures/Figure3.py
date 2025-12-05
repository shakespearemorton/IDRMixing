#!/usr/bin/env python3
"""Figure 3: Chemical grammar and demixing analysis (2x3 layout A-F)"""
import numpy as np
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import linregress
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed, dump

plt.rcParams.update({
    'font.size': 20, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 8, 'xtick.major.width': 2, 'ytick.major.size': 8, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

DATA_DIR = Path("figure3_data")
POSITIVE_AA, NEGATIVE_AA = set('KR'), set('DE')
N_SEEDS, TRAIN_SIZE, L1_RATIO, BEST_C = 1000, 0.8, 0.5, 0.1
STEMS = ['mean_lambda', 'scd', 'n_charged', 'N', 'faro', 'ncpr', 'kappa', 'shd']

SHORT_NAMES = {
    'mean_lambda_mean': r'$\overline{\hat{\lambda}}$', 'mean_lambda_abs_diff': r'$\Delta\hat{\lambda}$',
    'scd_mean': r'SCD$_{mean}$', 'scd_abs_diff': r'$\Delta$SCD',
    'n_charged_mean': r'Q#$_{mean}$', 'n_charged_abs_diff': r'$\Delta$Q#',
    'N_mean': r'$\overline{N}$', 'N_abs_diff': r'$\Delta$N',
    'faro_mean': r'F$_{aro,mean}$', 'faro_abs_diff': r'$\Delta$F$_{aro}$',
    'ncpr_mean': r'$\overline{NCPR}$', 'ncpr_abs_diff': r'$\Delta$NCPR',
    'kappa_mean': r'$\kappa_{mean}$', 'kappa_abs_diff': r'$\Delta\kappa$',
    'shd_mean': r'$\overline{SHD}$', 'shd_abs_diff': r'$\Delta$SHD',
}

GIN_CATEGORY_MAP = {
    7: 'C', 9: 'C', 18: 'C', 19: 'C', 23: 'C', 25: 'C', 26: 'C', 3: 'C', 8: 'C', 17: 'C', 24: 'C', 29: 'C',
    0: 'P', 1: 'P', 2: 'P', 4: 'P', 5: 'P', 6: 'P', 11: 'P', 12: 'P', 13: 'P', 14: 'P', 15: 'P', 16: 'P', 21: 'P', 22: 'P', 27: 'P',
    10: 'H', 20: 'H', 28: 'H'
}
INTERACTION_PALETTE = {'C-C': '#2E86AB', 'C-H': '#A23B72', 'C-P': '#F18F01', 'H-H': '#C73E1D', 'H-P': '#708238', 'P-P': '#1B998B'}

def save_csv(data_dict, filename):
    DATA_DIR.mkdir(exist_ok=True)
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 for v in data_dict.values())
    for k, v in data_dict.items():
        if not hasattr(v, '__len__') or isinstance(v, str): data_dict[k] = [v] * max_len
        elif len(v) < max_len: data_dict[k] = list(v) + [np.nan] * (max_len - len(v))
    pd.DataFrame(data_dict).to_csv(DATA_DIR / filename, index=False)

def get_interaction_type(gin1, gin2):
    if pd.isna(gin1) or pd.isna(gin2): return 'Other'
    cat1, cat2 = GIN_CATEGORY_MAP.get(int(gin1), 'Other'), GIN_CATEGORY_MAP.get(int(gin2), 'Other')
    return 'Other' if 'Other' in [cat1, cat2] else '-'.join(sorted([cat1, cat2]))

def calc_net_charge(seq):
    return sum(1 if aa in POSITIVE_AA else -1 for aa in str(seq).upper() if aa in POSITIVE_AA | NEGATIVE_AA) if pd.notna(seq) else 0

def calc_total_charged(seq):
    return sum(1 for aa in str(seq).upper() if aa in POSITIVE_AA | NEGATIVE_AA) if pd.notna(seq) else 0

def load_data():
    df_dem = pd.read_csv("../DATASETS/demixing.csv")
    df_prot = pd.read_csv("../DATASETS/gin_samples.csv")
    df_prot['n_charged'] = df_prot['fasta'].apply(lambda s: sum(1 if aa in 'KR' else -1 for aa in s if aa in 'KRED') if pd.notna(s) else np.nan)
    
    df = df_dem.copy()
    for p, suf in [('protein1', '_p1'), ('protein2', '_p2')]:
        df = pd.merge(df, df_prot, left_on=p, right_on='seq_name', how='left', suffixes=('', suf))
        df = df.drop(columns=['seq_name'])
        df = df.rename(columns={c: c + suf for c in df_prot.columns if c != 'seq_name' and c in df.columns and not c.endswith(('_p1', '_p2'))})
    return df

def build_features(df):
    X = pd.DataFrame(index=df.index)
    for s in STEMS:
        p1, p2 = f'{s}_p1', f'{s}_p2'
        if p1 in df.columns and p2 in df.columns:
            X[f'{s}_mean'] = (df[p1] + df[p2]) / 2
            X[f'{s}_abs_diff'] = (df[p1] - df[p2]).abs()
    return X.dropna()

def run_seed(seed, X, y, weights, groups, C):
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    try:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=L1_RATIO, C=C, max_iter=5000, random_state=42)
        model.fit(X_tr, (y.iloc[train_idx] > 0.5).astype(int), sample_weight=weights.iloc[train_idx])
        score = r2_score(y.iloc[test_idx], model.predict_proba(X_te)[:, 1], sample_weight=weights.iloc[test_idx])
        return dict(zip(X.columns, model.coef_[0])), score
    except: return None, None

def train_elastic_net(df):
    print("Training elastic net model...")
    df_filt = df[(df['composition1'] < 99) & (df['composition2'] < 99) & (df['dGij'] < -3)].copy()
    df_filt['pair_id'] = df_filt['protein1'] + "_" + df_filt['protein2']
    counts = df_filt['demixing_composition'].apply(lambda x: x > 0.5).value_counts()
    df_filt['weight'] = df_filt['demixing_composition'].apply(lambda x: 1.0 if x <= 0.5 else counts[False] / counts[True])
    
    X = build_features(df_filt)
    y, weights, groups = df_filt.loc[X.index, 'demixing_composition'], df_filt.loc[X.index, 'weight'], df_filt.loc[X.index, 'pair_id']
    
    results = Parallel(n_jobs=-1, verbose=1)(delayed(run_seed)(s, X, y, weights, groups, BEST_C) for s in range(N_SEEDS))
    
    all_coefs = {f: [r[0][f] for r in results if r[0]] for f in X.columns}
    coef_stats = {f: {'mean': np.mean(c), 'std': np.std(c), 'abs_mean': abs(np.mean(c))} for f, c in all_coefs.items() if c}
    top_features = sorted(coef_stats, key=lambda f: coef_stats[f]['abs_mean'], reverse=True)[:8]
    
    # Train final pipeline (scaler + model together for easy reuse)
    pipeline = Pipeline([('scaler', StandardScaler()), 
                         ('model', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=L1_RATIO, C=BEST_C, max_iter=5000, random_state=42))])
    X_top = X[top_features]
    pipeline.fit(X_top, (y > 0.5).astype(int), model__sample_weight=weights)
    
    # Compute raw coefficients for interpretability
    scaler, model = pipeline.named_steps['scaler'], pipeline.named_steps['model']
    raw_coefs = model.coef_[0] / scaler.scale_
    raw_intercept = model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_)
    
    # 50:50 subset for Panel A
    df_5050 = df_filt[(df_filt['composition1'] == 50) & (df_filt['composition2'] == 50)]
    X_5050 = build_features(df_5050)[top_features]
    
    return {'pipeline': pipeline, 'features': top_features, 'coef_stats': coef_stats,
            'raw_coefs': dict(zip(top_features, raw_coefs)), 'raw_intercept': raw_intercept,
            'X': X_5050, 'y': df_5050.loc[X_5050.index, 'demixing_composition']}

def save_model_for_reuse(model_data):
    """Save model with all info needed for prediction on new data."""
    DATA_DIR.mkdir(exist_ok=True)
    
    # Save sklearn pipeline (includes scaler)
    dump(model_data['pipeline'], DATA_DIR / "demixing_model_pipeline.joblib")
    
    # Save human-readable model specification
    spec = {
        'features_in_order': model_data['features'],
        'raw_intercept': model_data['raw_intercept'],
        'raw_coefficients': model_data['raw_coefs'],
        'feature_definitions': {
            f'{s}_mean': f"({s}_protein1 + {s}_protein2) / 2" for s in STEMS
        } | {f'{s}_abs_diff': f"abs({s}_protein1 - {s}_protein2)" for s in STEMS},
        'usage': "P(demixing) = sigmoid(intercept + sum(coef_i * feature_i))",
        'note': "For pipeline: load with joblib.load(), call pipeline.predict_proba(X[features])"
    }
    pd.DataFrame([spec]).to_json(DATA_DIR / "model_specification.json", orient='records', indent=2)
    
    # Save coefficients as CSV for easy viewing
    coef_df = pd.DataFrame({
        'feature': model_data['features'],
        'raw_coefficient': [model_data['raw_coefs'][f] for f in model_data['features']],
        'mean_coefficient': [model_data['coef_stats'][f]['mean'] for f in model_data['features']],
        'std_coefficient': [model_data['coef_stats'][f]['std'] for f in model_data['features']]
    })
    coef_df.to_csv(DATA_DIR / "model_coefficients.csv", index=False)
    print(f"Model saved to {DATA_DIR}/")

def prepare_panel_data(df):
    df_C = df[(df['composition1'] < 95) & (df['composition2'] < 95) & (df['dGij'] < -3)].copy()
    df_C['interaction_type'] = df_C.apply(lambda r: get_interaction_type(r['gin1'], r['gin2']), axis=1)
    df_C = df_C[df_C['interaction_type'] != 'Other'].drop_duplicates(subset=['protein1', 'protein2', 'composition1'])
    
    df_DE = df[(df['composition1'] < 95) & (df['composition2'] < 95) & (df['demixing_composition'] < 0.2)].copy()
    df_DE['dG_deviation'] = abs(df_DE['dG1'] - df_DE['dG2'])
    df_DE['average_dG'] = (df_DE['dG1'] + df_DE['dG2']) / 2
    df_DE = df_DE.dropna(subset=['dGij', 'dG_deviation', 'average_dG'])
    
    df_F = df[(df['composition1'] < 95) & (df['composition2'] < 95) & (df['demixing_composition'] < 0.2)].copy()
    df_F = df_F.dropna(subset=['fasta_p1', 'fasta_p2', 'nuse1', 'nuse2', 'dGij'])
    df_F['net_charge1'] = df_F['fasta_p1'].apply(calc_net_charge) * df_F['nuse1']
    df_F['net_charge2'] = df_F['fasta_p2'].apply(calc_net_charge) * df_F['nuse2']
    df_F['pair_id'] = df_F['protein1'] + '_' + df_F['protein2']
    df_F['abs_total_net_charge'] = (df_F['net_charge1'] + df_F['net_charge2']).abs()
    df_F['total_charged'] = (df_F['fasta_p1'].apply(calc_total_charged) * df_F['nuse1'] + 
                             df_F['fasta_p2'].apply(calc_total_charged) * df_F['nuse2'])
    pair_counts = df_F['pair_id'].value_counts()
    df_F = df_F[df_F['pair_id'].isin(pair_counts[pair_counts > 1].index)]
    
    return df_C, df_DE, df_F

def create_figure3():
    df = load_data()
    model_data = train_elastic_net(df)
    save_model_for_reuse(model_data)
    df_C, df_DE, df_F = prepare_panel_data(df)
    
    fig = plt.figure(figsize=(33, 22))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=-0.1, wspace=0.15, height_ratios=[0.5, 1])
    add_label = lambda ax, lbl: ax.text(-0.1, 1.12, lbl, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Panel A: Model fit
    ax = fig.add_subplot(gs[0, 0]); add_label(ax, 'A')
    pipeline, features = model_data['pipeline'], model_data['features']
    X, y = model_data['X'], model_data['y']
    X_scaled = pipeline.named_steps['scaler'].transform(X)
    model = pipeline.named_steps['model']
    raw_score = X_scaled.dot(model.coef_[0]) + model.intercept_[0]
    y_pred = model.predict_proba(X_scaled)[:, 1]
    sort_idx = np.argsort(raw_score)
    
    ax.scatter(raw_score, y, c='grey', alpha=0.6, s=80, zorder=2)
    ax.axhline(0.5, color='black', ls='--', lw=2.5, alpha=0.75, zorder=1)
    ax.axvline(0.0, color='grey', ls=':', lw=2.5, alpha=0.75, zorder=1)
    ax.plot(raw_score[sort_idx], y_pred[sort_idx], color='darkred', lw=5, ls='--', zorder=3, label='Model')
    ax.set_xlabel('Model raw score (logit)'); ax.set_ylabel('Demixing Index')
    ax.set_ylim(0, 1.1); ax.legend(fontsize=18, loc='upper left')
    ax.text(0.95, 0.05, f'n = {len(y)} (50:50)', transform=ax.transAxes, fontsize=18, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    save_csv({'raw_score': raw_score, 'predicted': y_pred, 'observed': y.values}, 'panelA_elastic_net.csv')
    
    # Panel B: Coefficients
    ax = fig.add_subplot(gs[0, 1]); add_label(ax, 'B')
    coef_stats = model_data['coef_stats']
    top_sorted = sorted(features, key=lambda f: coef_stats[f]['abs_mean'], reverse=True)
    abs_means = [coef_stats[f]['abs_mean'] for f in top_sorted]
    stds = [coef_stats[f]['std'] for f in top_sorted]
    colors = ["#629673" if coef_stats[f]['mean'] > 0 else "#2a2c2b" for f in top_sorted]
    
    ax.barh(range(len(top_sorted)), abs_means, xerr=stds, color=colors, alpha=0.8, edgecolor='black', lw=1.5, capsize=4)
    for i, (m, s, f) in enumerate(zip(abs_means, stds, top_sorted)):
        ax.text(m + s + 0.02, i, SHORT_NAMES.get(f, f), va='center', ha='left', fontsize=18)
    ax.set_yticks([]); ax.set_xlabel('|Mean Coefficient| ± SD')
    ax.set_xlim(0, max(m + s for m, s in zip(abs_means, stds)) * 1.4); ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#629673'), Patch(color='#2a2c2b')], ['→ Demixing', '→ Mixing'], loc='lower right', fontsize=18)
    save_csv({'feature': top_sorted, 'abs_mean': abs_means, 'std': stds, 'mean': [coef_stats[f]['mean'] for f in top_sorted]}, 'panelB_coefficients.csv')
    
    # Panel C: Interaction types
    ax = fig.add_subplot(gs[0, 2]); add_label(ax, 'C')
    order = df_C.groupby('interaction_type')['demixing_composition'].median().sort_values().index.tolist()
    bp = ax.boxplot([df_C[df_C['interaction_type'] == it]['demixing_composition'].values for it in order],
                    positions=range(len(order)), widths=0.6, patch_artist=True,
                    flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5, markersize=6),
                    boxprops=dict(lw=2), medianprops=dict(lw=2.5, color='white'), whiskerprops=dict(lw=2), capprops=dict(lw=2))
    for patch, it in zip(bp['boxes'], order): patch.set_facecolor(INTERACTION_PALETTE.get(it, 'lightgrey')); patch.set_alpha(0.8)
    ax.set_ylabel('Demixing Index'); ax.set_xlabel('Interaction Type')
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3, axis='y')
    for i, it in enumerate(order):
        ax.text(i, df_C[df_C['interaction_type'] == it]['demixing_composition'].max() + 0.05,
                f'n={len(df_C[df_C["interaction_type"] == it])}', ha='center', va='bottom', fontsize=18)
    save_csv({'interaction_type': df_C['interaction_type'].values, 'demixing': df_C['demixing_composition'].values}, 'panelC_interaction.csv')
    
    # Panel D: dGij vs |dG1 - dG2|
    ax = fig.add_subplot(gs[1, 0]); add_label(ax, 'D')
    h = sns.histplot(data=df_DE, x='dG_deviation', y='dGij', binwidth=(0.2, 0.2), cmap=cmaps.reds_light_r, cbar=True, ax=ax,
                     cbar_kws={'label': 'Count', 'shrink': 0.5, 'format': '%d'})
    from matplotlib.ticker import MaxNLocator
    h.collections[0].colorbar.locator = MaxNLocator(integer=True); h.collections[0].colorbar.update_ticks()
    ax.set_xlabel('|ΔG$_{ii}$ - ΔG$_{jj}$| ($k_BT$)'); ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)')
    ax.set_xlim(0, 10); ax.set_ylim(0.5, -10); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    save_csv({'dG_deviation': df_DE['dG_deviation'].values, 'dGij': df_DE['dGij'].values}, 'panelD_dG_diff.csv')
    
    # Panel E: dGij vs average dG
    ax = fig.add_subplot(gs[1, 1]); add_label(ax, 'E')
    sns.histplot(data=df_DE, x='average_dG', y='dGij', binwidth=(0.2, 0.2), cmap=cmaps.blues_light_r, cbar=True, ax=ax,
                 cbar_kws={'label': 'Count', 'shrink': 0.5, 'format': '%d'})
    slope, intercept, r_val, _, _ = linregress(df_DE['average_dG'], df_DE['dGij'])
    ax.plot([-10, 0], [slope * -10 + intercept, intercept], 'k--', lw=3)
    ax.plot([-10, 0], [-10, 0], color='grey', ls='--', lw=2)
    ax.text(0.95, 0.05, f'Slope = {slope:.2f}\n$R^2$ = {r_val**2:.2f}', transform=ax.transAxes, fontsize=18, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    ax.set_xlabel('(ΔG$_{ii}$ + ΔG$_{jj}$)/2 ($k_BT$)'); ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)')
    ax.set_xlim(0.5, -10); ax.set_ylim(0.5, -10); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    save_csv({'average_dG': df_DE['average_dG'].values, 'dGij': df_DE['dGij'].values, 'slope': slope, 'r_squared': r_val**2}, 'panelE_dG_avg.csv')
    
    # Panel F: dGij vs |Net Charge|
    ax = fig.add_subplot(gs[1, 2]); add_label(ax, 'F')
    for pid in df_F['pair_id'].unique():
        pdata = df_F[df_F['pair_id'] == pid].sort_values('abs_total_net_charge')
        ax.plot(pdata['abs_total_net_charge'], pdata['dGij'], color='gray', alpha=0.3, lw=1.5, zorder=1)
    scatter = ax.scatter(df_F['abs_total_net_charge'], df_F['dGij'], c=df_F['total_charged'], cmap='YlOrBr',
                         s=100, alpha=0.7, edgecolors='black', lw=1, zorder=2)
    ax.set_xlabel('|Net Charge in Simulation|'); ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)')
    ax.set_ylim(0.5, -10); ax.set_box_aspect(1); ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.5); cbar.set_label('Num. of Charged Residues', rotation=90, labelpad=20)
    save_csv({'abs_net_charge': df_F['abs_total_net_charge'].values, 'dGij': df_F['dGij'].values,
              'total_charged': df_F['total_charged'].values, 'pair_id': df_F['pair_id'].values}, 'panelF_charge.csv')
    
    plt.tight_layout()
    plt.savefig('Figure3.pdf', bbox_inches='tight')
    print("Figure 3 saved as Figure3.pdf")

if __name__ == '__main__':
    create_figure3()