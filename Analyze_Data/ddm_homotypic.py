import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import yaml
import sys

ATOMS_A3_TO_mM = 1.660539e6

def calc_zpatch(z, h, cutoff=0):
    """Find the largest contiguous region above cutoff."""
    ct, ct_max, zpatch, hpatch, zwin, hwin = 0., 0., [], [], [], []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x; zwin.append(z[ix]); hwin.append(x)
        else:
            if ct > ct_max: ct_max, zpatch, hpatch = ct, zwin, hwin
            ct, zwin, hwin = 0., [], []
    if ct > ct_max: zpatch, hpatch = zwin, hwin
    return np.array(zpatch), np.array(hpatch)

def analyze_3d_chunks(u, start=1000, chunk_size=30.0):
    """Compute average density per chunk over trajectory frames."""
    box = u.dimensions[:3]
    seq_len = len(u.segments[0].atoms)
    n_chunks = np.maximum(np.round(box / chunk_size).astype(int), 1)
    chunk_dims = box / n_chunks
    ny, nz = n_chunks[1], n_chunks[2]
    avg_counts = np.zeros(n_chunks, dtype=np.float64)
    
    ag = u.atoms
    lz = u.dimensions[2]
    edges = np.arange(0, lz + 1, 1)
    z = edges[:-1] + 0.5
    
    for i in range(len(u.trajectory) - start):
        ts = u.trajectory[i + start]
        
        # Center trajectory
        zpos = ag.positions[:, 2]
        h, _ = np.histogram(zpos, bins=edges)
        ag.translate([0, 0, -z[np.argmax(h)] + 0.5 * lz])
        ts = transformations.wrap(ag)(ts)
        
        zpos = ag.positions[:, 2]
        h, _ = np.histogram(zpos, bins=edges)
        zpatch, hpatch = calc_zpatch(z, h)
        ag.translate([0, 0, -np.average(zpatch, weights=hpatch) + 0.5 * lz])
        ts = transformations.wrap(ag)(ts)
        
        # Count atoms per chunk
        pos = ag.positions
        idx = np.clip((pos // chunk_dims).astype(int), 0, n_chunks - 1)
        flat_idx = idx[:, 0] * ny * nz + idx[:, 1] * nz + idx[:, 2]
        counts = np.bincount(flat_idx, minlength=np.prod(n_chunks)).reshape(n_chunks)
        avg_counts += (counts - avg_counts) / (i + 1)
    
    active_dens = avg_counts.ravel().reshape(-1, 1) / np.prod(chunk_dims) * ATOMS_A3_TO_mM / seq_len
    return active_dens

def process_directory(direc, S):
    """Process a single directory for chunk size S."""
    pdb = direc / "top.pdb"
    save_path = Path(f"results_{S}") / f"chunk_density_data_{direc.name}.npz"
    
    if not pdb.is_file() or save_path.exists():
        return
    
    dcd_files = list(direc.glob("*.dcd"))
    if not dcd_files:
        return
    
    # Get start frame from config
    config_path = direc / "config.yaml"
    start_frame = 1000
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        try:
            wfreq = config.get('runtime_settings', {}).get('wfreq')
            start_frame = int(1e9 / (10 * wfreq))
        except: pass
    
    u = mda.Universe(str(pdb), str(dcd_files[0]))
    print(f"Processing {direc.name}: {len(u.trajectory)} frames, starting at {start_frame}")
    
    active_data = analyze_3d_chunks(u, start=start_frame, chunk_size=S)
    print(f"Active data points: {len(active_data)} in {direc.name}")
    
    s = active_data[:, 0]
    try:
        kde = gaussian_kde(s)
    except:
        return
    
    vol_range = s.ptp()
    x_range = np.linspace(s.min() - 0.05 * vol_range, s.max() + 0.05 * vol_range, 4096)
    kde_y = kde(x_range)
    peaks, props = find_peaks(kde_y, height=1e-8)
    
    # Sort by position: dilute = lowest, dense = tallest of remaining
    sorted_idx = np.argsort(x_range[peaks])
    hist_dil = x_range[peaks[sorted_idx[0]]]
    hist_den = 0
    if len(peaks) > 1:
        remaining = sorted_idx[1:]
        hist_den = x_range[peaks[remaining[np.argmax(props['peak_heights'][remaining])]]]
    
    print(f"Dilute: {hist_dil:.2f} mM, Dense: {hist_den:.2f} mM")
    
    Path(f"results_{S}").mkdir(exist_ok=True)
    np.savez(save_path, active_data=active_data, kde_y=kde_y, x_range=x_range,
             peaks=peaks, hist_dil=hist_dil, hist_den=hist_den)

if __name__ == "__main__":
    job_index = int(sys.argv[1])
    directories = [p for p in Path.cwd().iterdir() if p.is_dir()]
    for S in [5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        process_directory(directories[job_index], S)