#!/usr/bin/env python3
"""
Supplementary Figure 2: Interaction type analysis
Analysis of heterotypic dG vs average dG for different GIN-based interaction types
(Charge-Charge, Charge-Hydrophobic, Charge-Polar, Hydrophobic-Hydrophobic, 
 Hydrophobic-Polar, Polar-Polar)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress

# ── Configuration ───────────────────────────────────────────────
CSV_INPUT_FILE = "../DATASETS/demixing.csv"
TIE_LINES_FILE = "../DATASETS/tie_lines.csv"
plt.rcParams.update({
    'font.size': 18,
    'axes.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'xtick.major.width': 2,
    'ytick.major.size': 6,
    'ytick.major.width': 2,
    'legend.frameon': False,
    'figure.dpi': 300
})

# ── Helper Functions ────────────────────────────────────────────
def save_filtering_criteria(criteria_dict, filename):
    """Save filtering criteria to text file."""
    from pathlib import Path
    output_dir = Path("supp_fig2_data")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        f.write(f"Filtering Criteria for Supplementary Figure 2\n")
        f.write("=" * 60 + "\n\n")
        for key, value in criteria_dict.items():
            f.write(f"{key}: {value}\n")

def save_data_to_csv(data_dict, filename):
    """Save data dictionary to CSV in supp_fig2_data directory."""
    from pathlib import Path
    output_dir = Path("supp_fig2_data")
    output_dir.mkdir(exist_ok=True)
    
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 
                  for v in data_dict.values())
    
    for key, val in data_dict.items():
        if not hasattr(val, '__len__') or isinstance(val, str):
            data_dict[key] = [val] * max_len
        elif len(val) < max_len:
            data_dict[key] = list(val) + [np.nan] * (max_len - len(val))
    
    pd.DataFrame(data_dict).to_csv(output_dir / filename, index=False)

# ── GIN Category Definitions ────────────────────────────────────
GIN_CATEGORY_MAP = {
    # Strongly Charged
    7: 'Charged', 9: 'Charged', 18: 'Charged',
    19: 'Charged', 23: 'Charged', 25: 'Charged',
    26: 'Charged', 3: 'Charged', 8: 'Charged',
    17: 'Charged', 24: 'Charged', 29: 'Charged',
    # Weakly Charged / Polar
    0: 'Polar', 1: 'Polar', 2: 'Polar',
    4: 'Polar', 5: 'Polar', 6: 'Polar',
    11: 'Polar', 12: 'Polar', 13: 'Polar',
    14: 'Polar', 15: 'Polar', 16: 'Polar',
    21: 'Polar', 22: 'Polar', 27: 'Polar',
    # Hydrophobic / Aromatic
    10: 'Hydrophobic', 20: 'Hydrophobic', 28: 'Hydrophobic'
}

INTERACTION_LABELS = {
    'Charged-Charged': 'Charge-Charge',
    'Charged-Hydrophobic': 'Charge-Hydrophobic',
    'Charged-Polar': 'Charge-Polar',
    'Hydrophobic-Hydrophobic': 'Hydrophobic-Hydrophobic',
    'Hydrophobic-Polar': 'Hydrophobic-Polar',
    'Polar-Polar': 'Polar-Polar'
}

# ── Helper Functions ────────────────────────────────────────────
def get_gin_category(gin_value):
    """Map a GIN number to its category."""
    if pd.isna(gin_value):
        return 'Other'
    return GIN_CATEGORY_MAP.get(int(gin_value), 'Other')

def get_interaction_type(gin1, gin2):
    """Get the interaction type from two GIN values."""
    if pd.isna(gin1) or pd.isna(gin2):
        return 'Other'
    cat1 = get_gin_category(gin1)
    cat2 = get_gin_category(gin2)
    if cat1 == 'Other' or cat2 == 'Other':
        return 'Other'
    # Sort to make (A,B) the same as (B,A)
    interaction = '-'.join(sorted([cat1, cat2]))
    return INTERACTION_LABELS.get(interaction, 'Other')

def calculate_dG_metrics(df_in):
    """Calculates all dG related metrics for a given dataframe."""
    df_out = df_in.copy()
    
    def get_dg_average(row):
        return (row['dG1'] + row['dG2']) / 2
    
    def extract_transfer_energy(row):
        if 'dGij' in row and pd.notna(row['dGij']):
            return row['dGij']
        return np.nan
    
    df_out['average_dG'] = df_out.apply(get_dg_average, axis=1)
    df_out['transfer_dG'] = df_out.apply(extract_transfer_energy, axis=1)
    
    return df_out.dropna(subset=['transfer_dG', 'average_dG'])

# ── Load and Merge Data ─────────────────────────────────────────
print("Loading data...")
df_demixing = pd.read_csv(CSV_INPUT_FILE)
df_tie_lines = pd.read_csv(TIE_LINES_FILE)

print("Merging datasets...")
df_merged = pd.merge(
    df_demixing,
    df_tie_lines[['protein1', 'protein2', 'composition1', 'composition2', 'phase_behavior']],
    left_on=['protein1', 'protein2', 'composition1', 'composition2'],
    right_on=['protein1', 'protein2', 'composition1', 'composition2'],
    how='left'
)

# Filter for mixed phase behavior only
print("Filtering for mixed phase behavior...")
df_mixed = df_merged[
    (df_merged['demixing_composition'] < 0.2) &
    (df_merged['composition1'] < 95) & 
    (df_merged['composition2'] < 95)
].copy()

# Calculate dG metrics
df_mixed = calculate_dG_metrics(df_mixed)
print(f"Total mixed phase systems: {len(df_mixed)}")

# Save filtering criteria
save_filtering_criteria({
    'Dataset': 'demixing.csv merged with tie_lines.csv',
    'Phase behavior filter': 'mixed only',
    'Composition1 filter': '< 95%',
    'Composition2 filter': '< 95%',
    'dG metrics': 'average_dG = (dG1 + dG2) / 2, transfer_dG = dGij',
    'Rows after filtering': len(df_mixed),
    'GIN categories used': 'Charged, Polar, Hydrophobic',
    'Interaction types': ', '.join(INTERACTION_LABELS.values()),
    'Other interactions': 'Excluded'
}, 'filtering_criteria.txt')

# ── Calculate Interaction Types ────────────────────────────────
print("Categorizing interactions...")
df_mixed['interaction_type'] = df_mixed.apply(
    lambda row: get_interaction_type(row['gin1'], row['gin2']), 
    axis=1
)

# Remove 'Other' category
df_mixed = df_mixed[df_mixed['interaction_type'] != 'Other']

# Count interactions by type
print("\nInteraction type counts:")
print(df_mixed['interaction_type'].value_counts())

# ── Create Figure with Subplots for Each Interaction Type ──────
interaction_types = sorted(df_mixed['interaction_type'].unique())
n_types = len(interaction_types)

# Create a 2x3 grid (adjust if needed)
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()

# Color palette for different interaction types
colors = sns.color_palette("husl", n_types)

# Print color information
print("\n" + "="*60)
print("COLOR ASSIGNMENTS FOR EACH INTERACTION TYPE")
print("="*60)
for idx, (interaction, color) in enumerate(zip(interaction_types, colors)):
    panel_label = chr(65 + idx)
    rgb = tuple(int(c * 255) for c in color)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
    print(f"Panel {panel_label} - {interaction}:")
    print(f"  RGB: {rgb}")
    print(f"  Hex: {hex_color}")
    print(f"  Float RGB: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
print("="*60 + "\n")

for idx, interaction in enumerate(interaction_types):
    ax = axes[idx]
    
    # Add panel label
    panel_label = chr(65 + idx)  # A, B, C, D, E, F...
    ax.text(-0.1, 1.2, panel_label, transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Filter data for this interaction type
    df_int = df_mixed[df_mixed['interaction_type'] == interaction]
    
    if len(df_int) < 3:  # Need at least 3 points for meaningful regression
        ax.text(0.5, 0.5, f'{interaction}\n(n={len(df_int)}, insufficient data)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0.5, -10)
        ax.set_ylim(0.5, -10)
        continue
    
    # Create 2D histogram
    sns.histplot(data=df_int, x='average_dG', y='transfer_dG',
                 binwidth=(0.3, 0.3), cbar=True, ax=ax, 
                 cbar_kws={'label': 'Count'},
                 color=colors[idx])
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        df_int['average_dG'], df_int['transfer_dG']
    )
    r_squared = r_value**2
    
    # Create line of best fit
    x_fit = np.array([-10, 0])
    y_fit = slope * x_fit + intercept
    
    # Plot the line
    ax.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=3, 
            label='Best Fit', zorder=10)
    
    # Plot diagonal reference line
    ax.plot([-10, 0], [-10, 0], color='grey', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add text with statistics
    text_label = f'Slope = {slope:.3f}\n$R^2$ = {r_squared:.3f}\nn = {len(df_int)}'
    ax.text(0.05, 0.95, text_label, 
            transform=ax.transAxes, 
            fontsize=16, 
            verticalalignment='top', 
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, edgecolor='black'))
    
    # Labels and formatting
    ax.set_xlabel('(ΔG$_{ii}$ + ΔG$_{jj}$)/2 ($k_BT$)')
    ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)')
    ax.set_title(interaction, fontsize=18, fontweight='bold')
    ax.set_xlim(0.5, -10)
    ax.set_ylim(0.5, -10)
    ax.grid(True, alpha=0.3)
    
    # Save data for this panel
    save_data_to_csv({
        'average_dG': df_int['average_dG'].values,
        'transfer_dG': df_int['transfer_dG'].values,
        'protein1': df_int['protein1'].values,
        'protein2': df_int['protein2'].values,
        'composition1': df_int['composition1'].values,
        'composition2': df_int['composition2'].values,
        'slope': slope,
        'r_squared': r_squared,
        'p_value': p_value
    }, f'panel{panel_label}_{interaction.replace("-", "_").replace(" ", "_")}_data.csv')
    
    print(f"\n{interaction}:")
    print(f"  n = {len(df_int)}")
    print(f"  Slope = {slope:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {p_value:.4e}")

# Hide unused subplots if any
for idx in range(n_types, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('Supplementary_Figure4.pdf', bbox_inches='tight', dpi=300)
print("\nSaved figure as 'Supplementary_Figure3.pdf'")
print("All data and filtering criteria saved in supp_fig3_data/ directory")

# ── Print Summary Statistics ────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY OF LINEAR FITS BY INTERACTION TYPE")
print("="*60)
print(f"{'Interaction Type':<30} {'n':<8} {'Slope':<10} {'R²':<10}")
print("-"*60)

for interaction in sorted(interaction_types):
    df_int = df_mixed[df_mixed['interaction_type'] == interaction]
    if len(df_int) >= 3:
        slope, intercept, r_value, p_value, std_err = linregress(
            df_int['average_dG'], df_int['transfer_dG']
        )
        r_squared = r_value**2
        print(f"{interaction:<30} {len(df_int):<8} {slope:<10.4f} {r_squared:<10.4f}")
    else:
        print(f"{interaction:<30} {len(df_int):<8} {'N/A':<10} {'N/A':<10}")

plt.show()