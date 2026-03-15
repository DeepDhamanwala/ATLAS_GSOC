import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

def main():
    print("Loading datasets")
    # 1. loadingg Data
    v_df = pd.read_csv('virtual_events.csv')
    r_df = pd.read_csv('real_events.csv')

    # we map real events in same space as virtual events
    r_df['pt'] = r_df['pt_real'] + r_df['z_gluon']
    r_df['y'] = r_df['y_real']

    # we combine both datasets into a single list of events (pt, y, w)
    events = pd.concat([v_df[['pt', 'y', 'weight']], r_df[['pt', 'y', 'weight']]], ignore_index=True)
    
    pt = events['pt'].values
    y = events['y'].values
    w = events['weight'].values.copy()
    
    # store original weights for comparision
    w_orig = w.copy()

    print(f"Total events: {len(events)}")
    print(f"Initial negative weight fraction: {np.sum(w[w < 0]) / np.sum(np.abs(w)):.4f}")

    # 3. implementing cell resampling
    # distance d is sqrt((pt_i - pt_j)**2 + 100 * (y_i - y_j)**2)
    # scaling the y-coordinates by sqrt(100) = 10, we can use a standard Euclidean KDTree
    y_scaled = y * 10.0
    coords = np.column_stack((pt, y_scaled))

    print("Building KD-Tree for spatial indexing")
    tree = KDTree(coords)

    negative_indices = np.where(w < 0)[0]
    print(f"Found {len(negative_indices)} negative weight seeds. Resampling in process")

    #building cells
    for seed_idx in negative_indices:
        # if seed was already absorbed by a previous cell and became positive, skip
        if w[seed_idx] >= 0:
            continue
            
        # querying nearest neighbors sorted by distance
        distances, indices = tree.query(coords[seed_idx], k=len(coords))
        
        cell_indices = []
        cell_weight_sum = 0.0
        
        # growing cell till weight is positive
        for idx in indices:
            cell_indices.append(idx)
            cell_weight_sum += w[idx]
            if cell_weight_sum >= 0:
                break
                
        # applying wt distro formula
        cell_indices = np.array(cell_indices)
        cell_w = w[cell_indices]
        
        sum_abs_w = np.sum(np.abs(cell_w))
        sum_w = np.sum(cell_w) # Equivalent to cell_weight_sum
        
        # w_i' = (|w_i| / sum(|w_j|)) * sum(w_j)
        new_w = (np.abs(cell_w) / sum_abs_w) * sum_w
        w[cell_indices] = new_w

    print(f"Final negative weight fraction: {np.sum(w[w < 0]) / np.sum(np.abs(w)):.4f}")

    # 4. plotting
    print("Generating validation histograms...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # pT distribution
    bins_pt = np.linspace(0, max(pt), 50)
    ax1.hist(pt, bins=bins_pt, weights=w_orig, alpha=0.6, label='Before Resampling', color='blue', edgecolor='black')
    ax1.hist(pt, bins=bins_pt, weights=w, alpha=0.6, label='After Resampling', color='orange', edgecolor='red', histtype='step', linewidth=2)
    ax1.set_title('Transverse Momentum ($p_T$) Distribution')
    ax1.set_xlabel('$p_T$ (GeV)')
    ax1.set_ylabel('Sum of Weights')
    ax1.legend()

    # y distribution
    bins_y = np.linspace(min(y), max(y), 50)
    ax2.hist(y, bins=bins_y, weights=w_orig, alpha=0.6, label='Before Resampling', color='blue', edgecolor='black')
    ax2.hist(y, bins=bins_y, weights=w, alpha=0.6, label='After Resampling', color='orange', edgecolor='red', histtype='step', linewidth=2)
    ax2.set_title('Rapidity ($y$) Distribution')
    ax2.set_xlabel('$y$')
    ax2.set_ylabel('Sum of Weights')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('kinematic_distributions.png')
    print("Saved 'kinematic_distributions.png'. Process complete.")

if __name__ == "__main__":
    main()
