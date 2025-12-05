#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
import openmm.unit as unit
import sys

# ——— Calvados helpers ————————————————————————————————
from sim_calvados import dual, run, fastadd, equil, build, run_temp

# ───────────────────────────────────────────────
# Global configuration
# ───────────────────────────────────────────────
TRAINING_CSV     = 'gin_redo.csv'          # sequence library

TEMP_PROD_K      = 293
BOX_PROD_NM      = 20          # cubic cross-section
BOC_PROD_LEN_NM  = 200
SAMPLING_PROD    = int(1e5)
RUNTIME_TEMP     = int(5e4)
RUNTIME_PROD     = int(2.5e8) # 2.5 us


# ───────────────────────────────────────────────
# Utility functions
# ───────────────────────────────────────────────

def run_simulation_slab(seq1, name1, n1, seq2, name2, n2, runtime, exp_tag):
    """
    Equilibrates two components, assembles them into a slab, and runs
    the production simulation.
    """
    print(f"\n--- Starting Simulation: {exp_tag} ---")
    print(f"Running for {runtime/1e6:.1f} ns with {n1} chains of {name1} and {n2} chains of {name2}")
    
    # 1. Staging-box equilibration for each component
    if Path(f'data/{exp_tag}.dcd').exists():
        print(f"Skipping equilibration for {exp_tag}, using existing data.")
        # Load existing positions
        u = mda.Universe(f'data/{exp_tag}.pdb', f'data/{exp_tag}.dcd')
        u.trajectory[-1]
        all_pos = u.atoms.positions/10
        exp_tag+='_2'
        if Path(f'data/{exp_tag}.dcd').exists():
            print(f"Skipping equilibration for {exp_tag}, using existing data.")
            # Load existing positions
            u = mda.Universe(f'data/{exp_tag}.pdb', f'data/{exp_tag}.dcd')
            u.trajectory[-1]
            all_pos = u.atoms.positions/10
            exp_tag+='_3'
        print(exp_tag)
    else:
        # Initialize positions for new simulation
        print('Starting New')
        total_chains = n1 + n2
        len1, len2 = len(seq1), len(seq2)
        total_points = n1 * len1 + n2 * len2
        
        all_pos = np.zeros((total_points, 3))
        
        # Generate random positions in center plane avoiding overlaps
        used_positions = set()
        current_idx = 0
        
        # Place chain 1 copies
        for chain in range(n1):
            # Find non-overlapping center position
            while True:
                center_x = np.random.uniform(-BOX_PROD_NM/4, BOX_PROD_NM/4)
                center_y = np.random.uniform(-BOX_PROD_NM/4, BOX_PROD_NM/4)
                pos_key = (round(center_x, 1), round(center_y, 1))
                if pos_key not in used_positions:
                    used_positions.add(pos_key)
                    break
            
            # Place chain points along Z-axis
            for i in range(len1):
                all_pos[current_idx] = [center_x, center_y, i * 0.38]
                current_idx += 1
        
        # Place chain 2 copies
        for chain in range(n2):
            # Find non-overlapping center position
            while True:
                center_x = np.random.uniform(-BOX_PROD_NM/4, BOX_PROD_NM/4)
                center_y = np.random.uniform(-BOX_PROD_NM/4, BOX_PROD_NM/4)
                pos_key = (round(center_x, 1), round(center_y, 1))
                if pos_key not in used_positions:
                    used_positions.add(pos_key)
                    break
            
            # Place chain points along Z-axis
            for i in range(len2):
                all_pos[current_idx] = [center_x, center_y, i * 0.38]
                current_idx += 1

        all_pos = equil(
            [seq1, seq2],
            [n1, n2],
            all_pos,
            400,
            [BOX_PROD_NM, BOX_PROD_NM, BOC_PROD_LEN_NM]
        )
        
    top, sys, _ = build(
        [seq1, seq2],
        [n1, n2],
        all_pos,
        TEMP_PROD_K,
        [BOX_PROD_NM, BOX_PROD_NM, BOC_PROD_LEN_NM],
        exp_tag
    )
    
    # 3. Final Production Run
    print(f'Starting final production run for {exp_tag}...')
    run_temp(top, sys, all_pos, TEMP_PROD_K, exp_tag,
             SAMPLING_PROD, RUNTIME_TEMP, runtime)
    print(f'✔ Simulation and analysis for {exp_tag} finished.')

# ───────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────

def main() -> None:
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Calvados affinity protocol for pair N in proteins.csv (using warmup data)"
    )
    parser.add_argument(
        "pair_index",
        type=int,
        help="0-based index into the list of unique unordered sequence pairs",
    )

    args = parser.parse_args()
    pair_index: int = args.pair_index

    # ------------------------------------------------------------------
    # 1 ─ load sequences and pick the requested pair
    # ------------------------------------------------------------------
    df_prot = pd.read_csv(TRAINING_CSV)

    row = df_prot.iloc[pair_index]
    seq1 = row["seq1"]
    seq2 = row["seq2"]
    name1 = row["seq_name1"]
    name2 = row["seq_name2"]
    nuse1 = row["nuse1"]
    nuse2 = row["nuse2"]
    rat1 = row['fraction_seq1']
    rat2 = row['fraction_seq2']

    exp_tag = f'{name1}_{name2}_{int(rat1 * 100)}_{int(rat2 * 100)}'
    run_simulation_slab(seq1, name1, nuse1, seq2, name2, nuse2,
                        RUNTIME_PROD, exp_tag)

    print('\n✔ simulation finished.')


# ───────────────────────────────────────────────
if __name__ == '__main__':
    main()

