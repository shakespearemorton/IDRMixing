#!/usr/bin/env python3
"""Supplementary Figure 4: Heterotypic ΔG vs average homotypic ΔG by interaction type"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

plt.rcParams.update({
    'font.size': 18, 'axes.linewidth': 2, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 6, 'xtick.major.width': 2, 'ytick.major.size': 6, 'ytick.major.width': 2,
    'legend.frameon': False, 'figure.dpi': 300
})

# GIN cluster to category mapping
GIN_CATEGORY = {
    **{k: 'Charged' for k in [3, 7, 8, 9, 17, 18, 19, 23, 24, 25, 26, 29]},
    **{k: 'Polar' for k in [0, 1, 2, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 27]},
    **{k: 'Hydrophobic' for k in [10, 20, 28]}
}

def get_interaction_type(gin1, gin2):
    if pd.isna(gin1) or pd.isna(gin2): return None
    cat1, cat2 = GIN_CATEGORY.get(int(gin1)), GIN_CATEGORY.get(int(gin2))
    if not cat1 or not cat2: return None
    return '-'.join(sorted([cat1, cat2]))

# Load and filter data
df = pd.read_csv("../DATASETS/demixing.csv")
df = df[(df['demixing_composition'] < 0.2) & (df['composition1'] < 95) & (df['composition2'] < 95)].copy()
df['avg_dG'] = (df['dG1'] + df['dG2']) / 2
df = df.dropna(subset=['dGij', 'avg_dG'])
df['interaction'] = df.apply(lambda r: get_interaction_type(r['gin1'], r['gin2']), axis=1)
df = df[df['interaction'].notna()]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
colors = sns.color_palette("husl", 6)
interactions = ['Charged-Charged', 'Charged-Hydrophobic', 'Charged-Polar', 
                'Hydrophobic-Hydrophobic', 'Hydrophobic-Polar', 'Polar-Polar']

for idx, (interaction, ax) in enumerate(zip(interactions, axes.flatten())):
    ax.text(-0.1, 1.2, chr(65 + idx), transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    df_int = df[df['interaction'] == interaction]
    
    if len(df_int) < 3:
        ax.text(0.5, 0.5, f'{interaction}\n(n={len(df_int)}, insufficient data)', 
                ha='center', va='center', transform=ax.transAxes)
    else:
        sns.histplot(data=df_int, x='avg_dG', y='dGij', binwidth=(0.3, 0.3), 
                     cbar=True, ax=ax, cbar_kws={'label': 'Count'}, color=colors[idx])
        
        # Linear regression
        slope, intercept, r, *_ = linregress(df_int['avg_dG'], df_int['dGij'])
        x_fit = np.array([-10, 0])
        ax.plot(x_fit, slope * x_fit + intercept, 'k--', lw=3, zorder=10)
        ax.plot([-10, 0], [-10, 0], color='grey', ls='--', lw=2, alpha=0.5)
        ax.text(0.05, 0.95, f'Slope = {slope:.2f}\n$R^2$ = {r**2:.2f}\nn = {len(df_int)}',
                transform=ax.transAxes, fontsize=16, va='top', 
                bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='black'))
    
    ax.set_xlabel('(ΔG$_{ii}$ + ΔG$_{jj}$)/2 ($k_BT$)')
    ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)')
    ax.set_title(interaction, fontsize=18, fontweight='bold')
    ax.set_xlim(0.5, -10); ax.set_ylim(0.5, -10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Supplementary_Figure4.pdf', bbox_inches='tight')
print("Saved Supplementary_Figure4.pdf")
plt.show()