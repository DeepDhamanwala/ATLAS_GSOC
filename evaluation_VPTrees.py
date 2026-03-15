import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VPTreeNode:
    def __init__(self, point, idx):
        self.point = point
        self.idx = idx
        self.median_dist = 0.0
        self.inside = None   # left child
        self.outside = None  # right child

# distance formula
def physics_distance(p1, p2):
    # p = [pt, y]
    return np.sqrt((p1[0] - p2[0])**2 + 100 * (p1[1] - p2[1])**2)

def build_vptree(points, indices):
    if len(points) == 0:
        return None
    
    # first point as the Vantage Point
    vp_point = points[0]
    vp_idx = indices[0]
    node = VPTreeNode(vp_point, vp_idx)
    
    if len(points) == 1:
        return node
    
    # calculating distances from the VP to all other points
    distances = np.array([physics_distance(vp_point, p) for p in points[1:]])
    
    # median distance to split the space into "inside" and "outside"
    node.median_dist = np.median(distances)
    
    # partition the remaining points
    inside_mask = distances < node.median_dist
    outside_mask = ~inside_mask
    
    # Recursively build subtrees
    node.inside = build_vptree(points[1:][inside_mask], indices[1:][inside_mask])
    node.outside = build_vptree(points[1:][outside_mask], indices[1:][outside_mask])
    
    return node

def search_vptree(node, target, k_neighbors, neighbors_list):
    if node is None:
        return
    
    # calculating dist from target to current vantage point
    dist = physics_distance(node.point, target)
    neighbors_list.append((dist, node.idx))
    
    # Determine which region the target falls into
    is_inside = dist < node.median_dist
    
    # search the preferred region first
    if is_inside:
        search_vptree(node.inside, target, k_neighbors, neighbors_list)
        # Only search the outside if the nearest points might cross the boundary
        search_vptree(node.outside, target, k_neighbors, neighbors_list)
    else:
        search_vptree(node.outside, target, k_neighbors, neighbors_list)
        search_vptree(node.inside, target, k_neighbors, neighbors_list)

def main():
    print("Loading datasets...")

    v_df = pd.read_csv('virtual_events.csv')
    r_df = pd.read_csv('real_events.csv')

    # mapping the real events into the same phase space as virtual events
    r_df['pt'] = r_df['pt_real'] + r_df['z_gluon']
    r_df['y'] = r_df['y_real']

    # combining both datasets into a single list of events (pt, y, w)
    events = pd.concat([v_df[['pt', 'y', 'weight']], r_df[['pt', 'y', 'weight']]], ignore_index=True)
    
    pt = events['pt'].values
    y = events['y'].values
    w = events['weight'].values.copy()
    
    # store original weights for the before/after histogram comparison
    w_orig = w.copy()

    print(f"Total events: {len(events)}")
    print(f"Initial negative weight fraction: {np.sum(w[w < 0]) / np.sum(np.abs(w)):.4f}")

    print("Building custom VP-Tree for spatial indexing...")

    coords_raw = np.column_stack((pt, y))
    all_indices = np.arange(len(coords_raw))

    #tree building
    vp_root = build_vptree(coords_raw, all_indices)

    negative_indices = np.where(w < 0)[0]
    print(f"Found {len(negative_indices)} negative weight seeds. Resampling...")

    for seed_idx in negative_indices:
        if w[seed_idx] >= 0:
            continue
            
        # searching VP-Tree for all neighbors and sort them by distance
        neighbors = []
        search_vptree(vp_root, coords_raw[seed_idx], len(coords_raw), neighbors)
        neighbors.sort(key=lambda x: x[0]) # Sort by distance
        
        cell_indices = []
        cell_weight_sum = 0.0
        
        # grow cell until net weight is positive
        for dist, idx in neighbors:
            cell_indices.append(idx)
            cell_weight_sum += w[idx]
            if cell_weight_sum >= 0:
                break
                
        # apply weight redistribution formula
        cell_indices = np.array(cell_indices)
        cell_w = w[cell_indices]
        
        sum_abs_w = np.sum(np.abs(cell_w))
        sum_w = np.sum(cell_w)
        
        new_w = (np.abs(cell_w) / sum_abs_w) * sum_w
        w[cell_indices] = new_w

    print(f"Final negative weight fraction: {np.sum(w[w < 0]) / np.sum(np.abs(w)):.4f}")

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
    plt.savefig('vp_tree_kinematic_distributions.png')
    print("Saved 'vp_tree_kinematic_distributions.png'. Process complete.")

if __name__ == "__main__":
    main()
