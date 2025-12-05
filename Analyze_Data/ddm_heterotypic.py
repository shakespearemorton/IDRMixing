import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
from tqdm import tqdm
from collections import defaultdict
from glob import glob
import pandas as pd
import os
import sys

# Constants
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

def load_fasta_mapping(csv_path):
    """Load FASTA to protein name mapping."""
    try:
        df = pd.read_csv(csv_path)
        return dict(zip(df['fasta'], df['seq_name']))
    except:
        return {}

def detect_species(u, fasta_map=None):
    """Detect species based on sequences."""
    fasta_map = fasta_map or {}
    seq_groups = defaultdict(list)
    
    for seg in u.segments:
        sig = tuple(res.name for res in seg.atoms)
        seq_groups[sig].append(seg)
    
    sigs = list(seq_groups)
    
    if len(sigs) == 1:
        sig = sigs[0]
        fasta = "".join([AA_MAP.get(c, 'X') for c in sig])
        name = fasta_map.get(fasta, "Species_1")
        atoms = u.atoms[[a.index for s in seq_groups[sig] for a in s.atoms]]
        return atoms, mda.AtomGroup([], u), name, ""
    
    elif len(sigs) == 2:
        sig1, sig2 = sigs
        fasta1 = "".join([AA_MAP.get(c, 'X') for c in sig1])
        fasta2 = "".join([AA_MAP.get(c, 'X') for c in sig2])
        name1 = fasta_map.get(fasta1, "Species_1")
        name2 = fasta_map.get(fasta2, "Species_2")
        
        atoms1 = u.atoms[[a.index for s in seq_groups[sig1] for a in s.atoms]]
        atoms2 = u.atoms[[a.index for s in seq_groups[sig2] for a in s.atoms]]
        return atoms1, atoms2, name1, name2
    
    else:
        print(f"More than 2 species ({len(sigs)}). Analyzing as single group.")
        return u.atoms, mda.AtomGroup([], u), "Mixed", ""

def calc_zpatch(z, h, cutoff=0):
    """Find the largest contiguous patch of density above cutoff."""
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = [] 
    hpatch = []
    
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    
    if ct > ct_max:  # edge case (slab at side of box)
        zpatch = zwindow
        hpatch = hwindow
    
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def analyze_3d_chunks(u, ag1, start=1000, chunk_size=35.0):
    """Analyze 3D density chunks with trajectory centering."""
    box = u.dimensions[:3]
    n_chunks = np.maximum(np.round(box / chunk_size).astype(int), 1)
    chunk_dims = box / n_chunks
    nx, ny, nz = n_chunks
    
    # Initialize arrays
    avg_counts_tot = np.zeros(n_chunks, dtype=np.float64)
    avg_counts_s1 = np.zeros(n_chunks, dtype=np.float64)
    s1_indices = ag1.indices
    
    # Setup for centering
    ag = u.atoms  # all atoms
    ag_ref = u.select_atoms('all')  # reference group for centering
    lz = u.dimensions[2]
    edges = np.arange(0, lz + 1, 1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    
    N_frames = len(u.trajectory)
    for i in range(N_frames - start):
        ts = u.trajectory[i + start]
        
        # Center trajectory following the protocol
        zpos = ag_ref.positions.T[2]
        h, e = np.histogram(zpos, bins=edges)
        zmax = z[np.argmax(h)]
        ag.translate(np.array([0, 0, -zmax + 0.5 * lz]))
        ts = transformations.wrap(ag)(ts)  # wrap
        
        zpos = ag_ref.positions.T[2]
        h, e = np.histogram(zpos, bins=edges)
        zpatch, hpatch = calc_zpatch(z, h)
        
        if len(zpatch) > 0:  # Only center if patch found
            zmid = np.average(zpatch, weights=hpatch)  # center of mass of slab
            ag.translate(np.array([0, 0, -zmid + 0.5 * lz]))
            ts = transformations.wrap(ag)(ts)  # wrap
        
        # Now get positions for chunk analysis
        pos = ag.positions
        
        # Total counts
        idx = np.clip(np.floor_divide(pos, chunk_dims).astype(int), 0, n_chunks - 1)
        flat_idx = idx[:, 0] * ny * nz + idx[:, 1] * nz + idx[:, 2]
        counts = np.bincount(flat_idx, minlength=nx*ny*nz).reshape(n_chunks)
        
        # Species 1 counts
        s1_pos = pos[s1_indices]
        s1_idx = np.clip(np.floor_divide(s1_pos, chunk_dims).astype(int), 0, n_chunks - 1)
        s1_flat = s1_idx[:, 0] * ny * nz + s1_idx[:, 1] * nz + s1_idx[:, 2]
        s1_counts = np.bincount(s1_flat, minlength=nx*ny*nz).reshape(n_chunks)
        
        # Update averages
        avg_counts_tot += (counts - avg_counts_tot) / (i + 1)
        avg_counts_s1 += (s1_counts - avg_counts_s1) / (i + 1)
    
    # Calculate fractions
    flat_tot = avg_counts_tot.ravel()
    flat_s1 = avg_counts_s1.ravel()
    frac_s1 = np.divide(flat_s1, flat_tot, out=np.zeros_like(flat_s1), where=flat_tot > 0)
    active_dens = flat_tot.reshape(-1, 1)
    
    # Get chunk centers
    ix, iy, iz = np.indices(n_chunks)
    centers = np.stack([
        ix * chunk_dims[0] + chunk_dims[0]/2,
        iy * chunk_dims[1] + chunk_dims[1]/2,
        iz * chunk_dims[2] + chunk_dims[2]/2
    ], axis=-1).reshape(-1, 3)
    
    data = np.hstack([active_dens, centers, frac_s1.reshape(-1,1)])
    
    return (data, chunk_dims)

def save_active_data(active_data, base_name, output_dir="active_data_final"):
    """Save active_data to numpy file."""
    os.makedirs(output_dir, exist_ok=True)
    active_data_file = os.path.join(output_dir, f"{base_name}_active_data.npy")
    np.save(active_data_file, active_data)
    return active_data_file

def check_active_data_exists(base_name, output_dir="active_data_final"):
    """Check if active_data already exists."""
    active_data_file = os.path.join(output_dir, f"{base_name}_active_data.npy")
    return os.path.exists(active_data_file)

def push_to_redo(df_row: pd.Series, n_frames: int, redo_path: str):
    """Append a single row to gin_redo.csv with total_frames recorded."""
    df_row = df_row.copy()
    df_row["total_frames"] = n_frames
    df_row.to_frame().T.to_csv(redo_path, mode="a", header=False, index=False)

if __name__ == "__main__":

    # ---------- I/O paths ----------------------------------------------------
    gin_prep   = pd.read_csv("gin_prep.csv")
    redo_path  = "gin_redo.csv"

    # Create the redo file once with a valid header (idempotent)
    if not os.path.exists(redo_path):
        (gin_prep.head(0)
            .assign(total_frames=pd.Series(dtype="int"))
            .to_csv(redo_path, index=False))

    # ---------- Pre-computed helpers ----------------------------------------
    csv_path        = "df_training_with_gin_groups.csv"
    fasta_map       = load_fasta_mapping(csv_path)
    name_to_fasta   = {v: k for k, v in fasta_map.items()}
    ATOMS_A3_TO_mM  = 1.660539e6

    # ---------- Determine which rows to process -----------------------------
    if len(sys.argv) > 1:
        # Process single row specified by index
        row_index = int(sys.argv[1])
        if not (0 <= row_index < len(gin_prep)):
            print(f"ERROR: row_index {row_index} is out of bounds. Must be 0 <= index < {len(gin_prep)}")
            sys.exit(1)
        rows_to_process = [row_index]
        print(f"--- Processing Row {row_index} ---")
    else:
        # Process all rows
        rows_to_process = range(len(gin_prep))
        print(f"--- Processing ALL {len(gin_prep)} rows ---")

    # ---------- Main Processing Loop ----------------------------------------
    for idx in rows_to_process:
        row = gin_prep.iloc[idx]
        
        seq_name1 = row['seq_name1']
        seq_name2 = row['seq_name2']
        frac1 = int(row['fraction_seq1'] * 100)
        frac2 = int(row['fraction_seq2'] * 100)
        
        # Construct base filename
        base_name = f"{seq_name1}_{seq_name2}_{frac1}_{frac2}"
        
        if len(sys.argv) <= 1:  # Only print if processing all
            print(f"\n=== Row {idx}/{len(gin_prep)-1}: {base_name} ===")
        
        # Check if already processed
        if check_active_data_exists(base_name):
            print(f"  Skipping (already processed)")
            continue
        
        # Find PDB and DCD files
        pdb_file = f"data/{base_name}.pdb"
        dcd_pattern = f"data/{base_name}*.dcd"
        dcd_files = sorted(glob(dcd_pattern), key=lambda x: len(x))
        
        # Validate files exist
        if not os.path.exists(pdb_file):
            print(f"  WARNING: PDB file not found: {pdb_file}")
            push_to_redo(row, 0, redo_path)
            continue
        
        if not dcd_files:
            print(f"  WARNING: No DCD files found matching: {dcd_pattern}")
            push_to_redo(row, 0, redo_path)
            continue
        
        print(f"  PDB: {os.path.basename(pdb_file)}")
        print(f"  DCDs: {[os.path.basename(d) for d in dcd_files]}")
        
        # Load trajectory
        try:
            u = (mda.Universe(pdb_file, dcd_files[0])
                 if len(dcd_files) == 1
                 else mda.Universe(pdb_file, dcd_files))
            n_frames = len(u.trajectory)
            print(f"  Total frames: {n_frames}")
        except (OSError, IOError) as e:
            print(f"  ERROR: Corrupted or unreadable DCD file: {e}")
            push_to_redo(row, -1, redo_path)
            continue
        except Exception as e:
            print(f"  ERROR: Unexpected error loading trajectory: {e}")
            push_to_redo(row, -1, redo_path)
            continue
        
        # Check minimum frames
        if n_frames < 4500:
            print(f"  WARNING: Insufficient frames ({n_frames} < 4500)")
            push_to_redo(row, n_frames, redo_path)
            continue
        
        # Analyze trajectory
        ag1, ag2, n1, n2 = detect_species(u, fasta_map)
        
        start_frame = 1000
        print(f"  Analyzing 3D chunks (start_frame={start_frame})...")
        active_data, chunk_dims = analyze_3d_chunks(
            u, ag1, start=start_frame, chunk_size=35
        )
        
        # Convert to mM
        chunk_vol = np.prod(chunk_dims)
        densities = active_data[:, 0] / chunk_vol
        active_data[:, 0] = densities * ATOMS_A3_TO_mM
        
        # Save results
        saved_file = save_active_data(active_data, base_name)
        print(f"  ✓ Successfully processed")
        print(f"  ✓ Saved to: {saved_file}")
    
    if len(sys.argv) <= 1:
        print(f"\n--- Completed ALL rows ---")
    else:
        print(f"--- Completed Row {rows_to_process[0]} ---")