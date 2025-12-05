
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import find_peaks, peak_widths
from scipy.stats import gaussian_kde
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.decomposition import PCA
from functools import lru_cache
import multiprocessing as mp

# ==================== Configuration ====================
DATA_DIR = "active_data_final"
OUTPUT_CSV = "../DATASETS/demixing.csv"
GIN_PREP_CSV = "../DATASETS/gin_prep.csv"
SAMPLES_CSV = "../DATASETS/gin_samples.csv"
GIN_SAMPLES_UPDATED_CSV = "../DATASETS/df_training.csv" 

# Numerical stability constant
EPSILON = 1e-9

def composition_basis_analysis(phi1, phi2, global_comp1, global_comp2, mask=None):

    if mask is not None:
        pts = np.column_stack((phi1[mask], phi2[mask]))
    else:
        pts = np.column_stack((phi1, phi2))

    if len(pts) < 10:
        return np.nan

    data_mean = np.mean(pts, axis=0)
    pts_centered = pts - data_mean
    global_comp_vector = np.array([global_comp1, global_comp2])
    norm_comp = np.linalg.norm(global_comp_vector)
    if norm_comp < EPSILON:
        return np.nan  

    # Vector 1 (u_comp): The "global composition unit vector"
    u_comp = global_comp_vector / norm_comp
    u_ortho = np.array([-u_comp[1], u_comp[0]])
    proj_comp = pts_centered @ u_comp
    proj_ortho = pts_centered @ u_ortho

    # 5. Calculate the variance ("eigenvalues") along each new axis
    var_comp = np.var(proj_comp)   # Variance along the composition axis
    var_ortho = np.var(proj_ortho) # Variance along the orthogonal (demixing) axis

    total_var = var_comp + var_ortho
    if total_var < EPSILON:
        return np.nan # No variance in the data

    demix_index = min(var_comp, var_ortho) / total_var

    return demix_index

def get_volume_fractions(active_data):
    """Calculate volume fractions from active data."""
    ATOMS_A3_TO_mM = 1.660539e6
    CHUNK_SIZE = 35
    box = np.asarray([200, 200, 2000])
    n_chunks = np.maximum(np.round(box / CHUNK_SIZE).astype(int), 1)
    chunk_dims = box / n_chunks
    chunk_volume = np.prod(chunk_dims)
    
    fraction_s1 = active_data[:, 4]
    d = chunk_volume * active_data[:, 0] / ATOMS_A3_TO_mM
    d1 = d * fraction_s1
    d2 = d * (1 - fraction_s1)
    num_chunks = np.prod(n_chunks)
    d = np.pad(d, (0, num_chunks - len(d)), mode='constant')
    d1 = np.pad(d1, (0, num_chunks - len(d1)), mode='constant')
    d2 = np.pad(d2, (0, num_chunks - len(d2)), mode='constant')
    return d,d1,d2

def peak_finding(d):
    try:
        kde = gaussian_kde(d)
    except:
        return (np.nan, np.nan, np.nan, np.array([]), np.array([]), np.nan, np.array([]), np.nan, np.nan)
    
    vol_range = d.ptp()
    x_range = np.linspace(d.min() - 0.05 * vol_range, 
                          d.max() + 0.05 * vol_range, 4096)
    kde_y = kde(x_range)
    peaks, properties = find_peaks(kde_y,height = 1e-9)
    peak_positions = x_range[peaks]
    peak_heights = properties['peak_heights']
    
    # Sort peaks by position (lowest concentration first = dilute)
    sorted_indices = np.argsort(peak_positions)
    c_dil = max(0,x_range[peaks[sorted_indices[0]]])
    
    # Exclude the dilute peak and find the tallest remaining peak
    if len(peaks) > 1:
        remaining_indices = sorted_indices[1:]  # Exclude the first (dilute) peak
        remaining_heights = peak_heights[remaining_indices]
        tallest_remaining = remaining_indices[np.argmax(remaining_heights)]
        c_den = peak_positions[tallest_remaining]
        hetero_dg = np.log(c_dil/(c_den + EPSILON))
    else:
        c_den = 0
        hetero_dg = 0
    
    
    peak_xs = x_range[peaks]
    peak_ys = kde_y[peaks]
    # Initialize outputs
    dense_phase_frac_s1 = np.nan
    
    # Multi-peak analysis (phase-separated systems)
    dilute_width = np.nan
    dense_width = np.nan
    if len(peaks) > 1:        
        widths = peak_widths(kde_y, peaks, rel_height=1)[0]
        dense_peak_idx = tallest_remaining
        dilute_peak_idx = sorted_indices[0]
        dense_width = widths[dense_peak_idx]
        dilute_width = widths[dilute_peak_idx]
    return (c_dil, c_den, hetero_dg, peak_xs, peak_ys, dense_phase_frac_s1, peaks, dilute_width, dense_width)

# ==================== File Processing ====================
def process_single_file(args):

    (file_path, s1, s2,sl1,sl2, n1,n2,seq1,seq2,
     dG1, dG2, cden1, cden2, cdil1, cdil2, gin1, gin2) = args
    
    # Parse composition from filename
    basename = os.path.basename(file_path)
    comp_str = basename.replace(f"{s1}_{s2}_", "").replace("_active_data.npy", "")
    try:
        comp1, comp2 = map(int, comp_str.split('_'))
    except:
        return None
    

    tot_res = n1 * sl1 + n2 * sl2
    phi1_g = (n1 * sl1) / tot_res
    phi2_g = (n2 * sl2) / tot_res
    # Load trajectory data
    data = np.load(file_path)
    d,d1,d2 = get_volume_fractions(data)
    demix_composition = composition_basis_analysis(d1, d2, phi1_g, phi2_g)
    # Phase identification via KDE peak finding
    (c_dil, c_den, hetero_dg, peak_xs, peak_ys, dense_phase_frac_s1, peaks, dilute_width, dense_width) = peak_finding(d)
    (c_dil1, c_den1, _, _, _, _, _, _, _) = peak_finding(d1)
    (c_dil2, c_den2, _, _, _, _, _, _, _) = peak_finding(d2)

    
    
    return {
        'protein1': s1, 'protein2': s2, 
        'dG1': dG1, 'dG2': dG2,
        'composition1': comp1, 'composition2': comp2,
        'nuse1': n1, 'nuse2': n2,
        'seq1': seq1, 'seq2': seq2,
        'phi1_global': phi1_g, 'phi2_global': phi2_g,
        'dense_phase_fraction_s1': dense_phase_frac_s1,
        'mix_cdil': c_dil, 'mix_cden': c_den,
        'hetero_cdil1': c_dil1, 'hetero_cden1': c_den1,
        'hetero_cdil2': c_dil2, 'hetero_cden2': c_den2,
        'number_of_peaks': len(peaks), 'peak_positions': peak_xs.tolist(), 
        'peak_heights': peak_ys.tolist(),
        'dilute_peak_width': dilute_width if len(peaks) > 1 else np.nan,
        'dense_peak_width': dense_width if len(peaks) > 1 else np.nan,
        'demixing_composition': 2*demix_composition,
        'dGij': hetero_dg, 'gin1': gin1, 'gin2': gin2,
        'cden1': cden1, 'cden2': cden2, 'cdil1': cdil1, 'cdil2': cdil2,
    }


# ==================== Main Workflow ====================
def main():
    """
    Main analysis pipeline with parallel processing.
    
    Processes all simulation trajectories, computes demixing metrics,
    and outputs comprehensive results CSV.
    """
    # Load metadata
    try:
        df_prep = pd.read_csv(GIN_PREP_CSV)
        df_samples = pd.read_csv(SAMPLES_CSV).set_index('seq_name')
        df_gin = pd.read_csv(GIN_SAMPLES_UPDATED_CSV).set_index('seq_name')
    except FileNotFoundError as e:
        print(f"Error: Could not find required input file. {e}")
        return

    
    # Prepare all processing tasks
    tasks = []
    for _, row in df_prep.iterrows():
        s1, s2 = row['seq_name1'], row['seq_name2']
        c1, c2 = row['fraction_seq1'], row['fraction_seq2']
        n1, n2 = row['nuse1'], row['nuse2']
        files = glob(os.path.join(DATA_DIR, f"{s1}_{s2}_{int(c1*100)}_{int(c2*100)}_active_data.npy"))
        
        if not files:
            continue
        
        # Pre-calculate protein properties
        seq1, seq2 = df_samples.at[s1, 'fasta'], df_samples.at[s2, 'fasta']
        gin1,gin2 = df_gin.at[s1, 'GIN_group'], df_gin.at[s2, 'GIN_group']
        
        for fp in files:
            tasks.append((fp, s1, s2, len(seq1), len(seq2), 
                         n1,n2,seq1,seq2,
                         df_gin.at[s1, 'domain_dG'], df_gin.at[s2, 'domain_dG'],
                         df_samples.at[s1, 'cden'], df_samples.at[s2, 'cden'],
                         df_samples.at[s1, 'cdil'], df_samples.at[s2, 'cdil'],
                          gin1, gin2))
    
    # Parallel processing
    n_workers = 1#max(1, mp.cpu_count() - 2)
    results = []
    
    if n_workers > 1:
        print(f"Processing {len(tasks)} simulations with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_file, t): t for t in tasks}
            
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
    else:
        print(f"Processing {len(tasks)} simulations sequentially...")
        for t in tasks:
            res = process_single_file(t)
            if res is not None:
                results.append(res)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')

if __name__ == "__main__":
    main()