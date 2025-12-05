import pandas as pd
from typing import List, Dict, Set, Tuple

def unique_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(len(df) - 1) for j in range(i + 1, len(df))]

def is_valid_composition(nuse1: int, nuse2: int) -> bool:
    """Reject 100-0 or 0-100 compositions"""
    return nuse1 > 0 and nuse2 > 0

def find_best_n1(target_total: int, L1: int, L2: int, n1_ideal: float, 
                 exclude: Set[int] = None) -> Tuple[int, int]:
    """Find best integer n1, n2 satisfying constraints"""
    exclude = exclude or set()
    best, min_error = (-1, -1), float('inf')
    
    for n1 in range(max(1, int(target_total / L1) - 5), int(target_total / L1) + 6):
        remainder = target_total - n1 * L1
        if remainder > 0 and remainder % L2 == 0:
            n2 = remainder // L2
            if n2 > 0 and n1 not in exclude:
                error = abs(n1 - n1_ideal)
                if error < min_error:
                    min_error, best = error, (n1, n2)
    return best

def generate_compositions(L1: int, L2: int, guideline_total: int = 36000, 
                          target_ratios: List[float] = None) -> List[Dict]:
    """Generate diverse compositions for a sequence pair"""
    target_ratios = target_ratios or [0.9, 0.75, 0.5, 0.25, 0.1]
    total_mults = [1.0, 0.85, 0.95, 1.05, 1.15, 1.25]
    all_ratios = target_ratios + [0.95, 0.85, 0.65, 0.35, 0.15, 0.05]
    
    compositions, used_n1 = [], set()
    
    for mult in total_mults:
        adj_total = int(guideline_total * mult)
        n1_ref, n2_ref = round((adj_total * 0.5) / L1), round((adj_total * 0.5) / L2)
        target_total = n1_ref * L1 + n2_ref * L2
        
        for r in all_ratios:
            if len(compositions) >= 5:
                break
            n1_ideal = (r * target_total) / L1
            n1, n2 = find_best_n1(target_total, L1, L2, n1_ideal, used_n1)
            
            if n1 != -1 and is_valid_composition(n1, n2):
                total_aa = n1 * L1 + n2 * L2
                compositions.append({
                    'nuse1': n1, 'nuse2': n2, 'total_aa': total_aa,
                    'fraction_seq1': round(n1 * L1 / total_aa, 2),
                    'fraction_seq2': round(n2 * L2 / total_aa, 2)
                })
                used_n1.add(n1)
        if len(compositions) >= 5:
            break
    
    # Select 5 most diverse by fraction_seq1
    if len(compositions) > 5:
        compositions.sort(key=lambda x: x['fraction_seq1'])
        n = len(compositions)
        indices = [0, n-1, n//2, n//4, 3*n//4]
        seen, selected = set(), []
        for idx in indices:
            c = compositions[idx]
            if c['nuse1'] not in seen:
                selected.append(c)
                seen.add(c['nuse1'])
        compositions = selected[:5]
    
    # Sort and assign composition indices
    compositions.sort(key=lambda x: x['fraction_seq1'])
    for i, c in enumerate(compositions):
        c['composition'] = i
    return compositions

def main() -> None:
    df_prot = pd.read_csv("gin_samples.csv")
    pair_list = unique_pairs(df_prot)
    print(f"Found {len(pair_list)} unique pairs in gin_samples.csv")
    
    results, recovered, failed = [], [], []
    
    for idx, (i, j) in enumerate(pair_list):
        row1, row2 = df_prot.iloc[i], df_prot.iloc[j]
        seq1, seq2, name1, name2 = row1["fasta"], row2["fasta"], row1["seq_name"], row2["seq_name"]
        
        comps = generate_compositions(len(seq1), len(seq2))
        
        for c in comps:
            c.update({'seq_name1': name1, 'seq_name2': name2, 'seq1': seq1, 'seq2': seq2})
        
        if len(comps) == 5:
            results.extend(comps)
        elif len(comps) > 0:
            recovered.extend(comps)
            print(f"Pair {idx + 1}: Only {len(comps)} valid compositions")
        else:
            failed.append({'pair_idx': idx, 'seq_name1': name1, 'seq_name2': name2})
            print(f"Pair {idx + 1}: No valid compositions")
    
    pd.DataFrame(results).to_csv('../DATASETS/gin_prep.csv', index=False)
    pd.DataFrame(recovered).to_csv('../DATASETS/gin_prep_recovered.csv', index=False)
    pd.DataFrame(failed).to_csv('../DATASETS/gin_prep_failed.csv', index=False)
    
    print(f"\nResults summary:")
    print(f"  Complete pairs (5 compositions): {len(results) // 5} -> gin_prep.csv")
    print(f"  Partial pairs: {len(recovered)} compositions -> gin_prep_recovered.csv")
    print(f"  Failed pairs: {len(failed)} -> gin_prep_failed.csv")

if __name__ == "__main__":
    main()