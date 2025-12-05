import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('../DATASETS/df_training.csv')


clipped_data = data[data['N'] < 250]
valid_groups = clipped_data.groupby('GIN_group').filter(lambda group: (group['dG'] < -5).any())
ignored_groups = clipped_data.groupby('GIN_group').filter(lambda group: not (group['dG'] < -5).any())

def select_shortest_sequence(group, dG_range):
    range_group = group[(group['dG'] >= dG_range[0]) & (group['dG'] < dG_range[1])]
    if not range_group.empty:
        shortest_sequence = range_group.loc[range_group['N'].idxmin()]
        return shortest_sequence
    else:
        return None
    
dG_ranges = [(-10, -8), (-8, -6), (-6, -4), (-4, -2)]
selected_entries = []

for gin_group, group in valid_groups.groupby('GIN_group'):
    for dG_range in dG_ranges:
        shortest = select_shortest_sequence(group, dG_range)
        if shortest is not None:
            selected_entries.append(shortest)

ignored_entries = []
for gin_group, group in ignored_groups.groupby('GIN_group'):
    group_filtered = group[group['N'] < 100]
    if not group_filtered.empty:
        selected_random = group_filtered.sample(n=1)
        ignored_entries.append(selected_random)
final_selected_entries = selected_entries + [entry.iloc[0] for entry in ignored_entries]
final_selected_df = pd.DataFrame(final_selected_entries)
final_selected_df.to_csv('../DATASETS/gin_samples.csv', index=False)
print("File saved as 'gin_samples.csv'")
