# Cell Resampling for Negative Weight Mitigation
This repository contains the solution for the GSoC Task "Negative Weight Mitigation with Cell Resampling."

## Structure:
1. **Data Pipeline & Born Projection:** The script parses real_events.csv and virtual_events.csv. Real events are mapped to the born phase space by shifting their transverse momentum ($p_T = p_{T,real} + z_{gluon}$) , allowing both datasets to be evaluated in the same dimensionality.

2. **Spatial Indexing & The VP-Tree Connection:** To perform the cell resampling efficiently, the phase space $(p_T, y)$ is loaded into a spatial index. While I am aware that the  CRES Rust codebase uses Vantage-Point Trees algorithm to  handle arbitrary kinematic distance metrics, I used scipy.spatial.KDTree for this in order to leverage its highly optimized C-backend.
I have already tried to implement VP tree and both algorithms resulted in same outputs.
To utilize standard Euclidean queries while using the requested distance formula provided ( $d_{i,j} = \sqrt{\Delta p_T^2 + 100 \Delta y^2}$ ), the component is pre-scaled by a factor of 10 prior to tree construction. This will provide the necessary $\mathcal{O}(N \log N)$ search speeds while   keeping the prototype lean.

4. **Dynamic Cell Growth:** The algorithm iterates over negatively weighted events aka seeeds. For each seed, it queries the KD Tree for nearest neighbors, aggregating them until the localized cell achieves a positive weight ($\ge 0$).

5. **Redistribution:** Once a cell is purely positive, the absolute weight fraction formula is applied in a vectorized manner to redistribute the event weights, squashing all localized negative values to zero while perfectly preserving the aggregate cross-section.

6. **Computational Complexity:** Let $N$ be the total number of events and $M$ be the number of initial negative weight seeds.\
   Building the KD-Tree: $\mathcal{O}(N \log N)$ space and time.\
   Nearest-Neighbor Search: For each seed, querying neighbors dynamically scales as $\mathcal{O}(K \log N)$, where $K$ is the number of events required to reach a positive sum.<br>
   In worst-case scenario (e.g., entire data contains negative weights), querying all neighbors is $\mathcal{O}(N \log N)$.\
   Overall Time Complexity: The worst-case runtime is bounded by $\mathcal{O}(M \cdot N \log N)$, average-case performance is lower since $\bar{K} \ll N$.

## Discussion Question
### Why is this scaling factor necessary? 

1. We use the factor of 100 as it's a perfect square, values like 98 or 99 can also be chosen which will give similar results but would throw in extra decimal values.

2. The rapidity angle **y** is often tiny so is traverse momentum **pT**. If we don't scale **y**, the **pT** difference dominates the distance calculation. Since both are in single digit decimal values multiplying them by 10 brings both in same order.

### What would happen to the physical distributions if we just used standard Euclidean distance without scaling?

If we directly used standard Euclidean distance without scaling, the nearest neighbour would improperly cluster events having different rapidities just because their **pT** values are same.

### How does the limit of infinite generated events affect this choice?

Each event will have nearest neighbors located very very closely in both variables simultaneously. As the geometric distance between neighbours would be almost **0** the variable **pT** would no longer dominate over **y**. The cells would ultimately shrink to pointlike size which may ensure good localized unweighting.
