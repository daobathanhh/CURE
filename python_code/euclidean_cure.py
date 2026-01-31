"""
CURE Algorithm - Euclidean Distance Version

Implementation based on the paper:
"CURE: An Efficient Clustering Algorithm for Large Databases"
by Sudipto Guha, Rajeev Rastogi, and Kyuseok Shim (SIGMOD 1998)

Contains:
- CURE: Base hierarchical clustering algorithm
- Scalable_CURE: Optimized version for large datasets using sampling and partitioning
"""

import numpy as np
import heapq
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Dict, Any
import warnings


# ============================================================
# CLUSTER CLASS
# ============================================================

class Cluster:
    """
    Represents a cluster in the CURE algorithm.
    
    Attributes:
        id: Unique cluster identifier
        points_idx: List of original point indices in this cluster
        mean: Centroid of the cluster
        reps: List of representative points (after shrinking)
        alive: Whether cluster is still active
        dist: Distance to closest cluster
        closest: ID of closest cluster
    """
    
    def __init__(self, cluster_id: int, points_idx: List[int], reps: List[np.ndarray], mean: np.ndarray):
        self.id = cluster_id
        self.points_idx = points_idx if isinstance(points_idx, list) else [points_idx]
        self.mean = mean
        self.reps = reps if isinstance(reps, list) else [reps]
        self.alive = True
        self.dist = float('inf')
        self.closest: Optional[int] = None
    
    def __lt__(self, other):
        """For heap comparison - compare by distance to closest."""
        return self.dist < other.dist
    
    def to_heap_entry(self) -> Tuple[float, int, 'Cluster']:
        """Create heap entry tuple."""
        return (self.dist, self.id, self)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(p - q)


def cluster_distance(u: Cluster, v: Cluster) -> float:
    """
    Compute distance between two clusters.
    
    Distance = minimum Euclidean distance between any pair of representatives.
    Uses vectorized computation for efficiency.
    """
    if len(u.reps) == 0 or len(v.reps) == 0:
        return float('inf')
    
    u_reps = np.array(u.reps)
    v_reps = np.array(v.reps)
    
    # Compute all pairwise distances at once
    dists = cdist(u_reps, v_reps, metric='euclidean')
    return np.min(dists)


# ============================================================
# CURE CLASS - BASE IMPLEMENTATION
# ============================================================

class CURE:
    """
    CURE (Clustering Using REpresentatives) Algorithm - Euclidean Distance.
    
    Based on Figure 5 and Figure 6 from the CURE paper.
    
    The algorithm:
    1. Each point starts as its own cluster
    2. Build KD-tree T with all representative points
    3. Build heap Q ordered by distance to closest cluster
    4. While more than k clusters remain:
       - Extract cluster u with minimum distance to its closest
       - Merge u with v (u's closest cluster)
       - Update KD-tree and heap
       - Update closest pointers for affected clusters
    
    Attributes:
        k: Number of clusters to form
        c: Number of representative points per cluster
        alpha: Shrink factor (0 to 1)
    """
    
    def __init__(self, k: int = 5, c: int = 5, alpha: float = 0.3):
        """
        Initialize CURE algorithm.
        
        Args:
            k: Number of clusters to form
            c: Number of representative points per cluster
            alpha: Shrink factor - representatives are shrunk toward centroid by this fraction
                   alpha=0: No shrinking (like MST/single-link)
                   alpha=1: Full shrinking (like centroid method)
                   Recommended: 0.2-0.7 for non-spherical clusters
        """
        self.k = int(k)
        self.c = int(c)
        self.alpha = float(alpha)
        
        # Set during fit
        self.S: Optional[np.ndarray] = None
        self.n: int = 0
        self.d: int = 0
        self.clusters_: List[Cluster] = []
        self.labels_: Optional[np.ndarray] = None
    
    def _build_kd_tree(self, clusters: Dict[int, Cluster]) -> Tuple[Optional[cKDTree], Dict[int, int]]:
        """
        Build KD-tree from representative points of all active clusters.
        
        Returns:
            Tuple of (KD-tree, rep_point_to_cluster_id mapping)
        """
        rep_points = []
        rep_map = {}  # index in rep_points -> cluster_id
        
        for cluster_id, cluster in clusters.items():
            if not cluster.alive:
                continue
            for rep in cluster.reps:
                rep_map[len(rep_points)] = cluster_id
                rep_points.append(rep)
        
        if not rep_points:
            return None, {}
        
        return cKDTree(np.array(rep_points)), rep_map
    
    def _find_closest_cluster_kdtree(
        self, 
        query_cluster: Cluster, 
        T: cKDTree, 
        rep_map: Dict[int, int],
        threshold: float = float('inf')
    ) -> Tuple[Optional[int], float]:
        """
        Find closest cluster using KD-tree (Step 15 in paper).
        
        Args:
            query_cluster: Cluster to find neighbor for
            T: KD-tree of representative points
            rep_map: Mapping from rep index to cluster id
            threshold: Maximum distance to consider
            
        Returns:
            (closest_cluster_id, distance) or (None, inf)
        """
        if T is None:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_id = None
        
        for query_rep in query_cluster.reps:
            # Query for multiple neighbors to ensure we don't just get our own cluster
            k = min(self.c * 2 + 2, len(T.data))
            distances, indices = T.query(query_rep, k=k, distance_upper_bound=threshold)
            
            if np.isscalar(distances):
                distances = [distances]
                indices = [indices]
            
            for dist, idx in zip(distances, indices):
                if dist >= threshold or idx >= len(T.data):
                    continue
                
                neighbor_cluster_id = rep_map.get(idx)
                if neighbor_cluster_id is None or neighbor_cluster_id == query_cluster.id:
                    continue
                
                if dist < min_dist:
                    min_dist = dist
                    closest_id = neighbor_cluster_id
        
        return closest_id, min_dist
    
    def _find_closest_cluster_brute(
        self, 
        query_cluster: Cluster, 
        clusters: Dict[int, Cluster]
    ) -> Tuple[Optional[int], float]:
        """
        Find closest cluster using brute force.
        
        Args:
            query_cluster: Cluster to find neighbor for
            clusters: All clusters
            
        Returns:
            (closest_cluster_id, distance) or (None, inf)
        """
        min_dist = float('inf')
        closest_id = None
        
        for cluster_id, cluster in clusters.items():
            if cluster_id == query_cluster.id or not cluster.alive:
                continue
            
            dist = cluster_distance(query_cluster, cluster)
            if dist < min_dist:
                min_dist = dist
                closest_id = cluster_id
        
        return closest_id, min_dist
    
    def _unshrink_point(self, shrunk_point: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """
        Recover original point from shrunk representative.
        
        shrunk = original + alpha * (mean - original)
        original = (shrunk - alpha * mean) / (1 - alpha)
        """
        if self.alpha == 1.0:
            return mean.copy()
        return (shrunk_point - self.alpha * mean) / (1.0 - self.alpha)
    
    def _merge_clusters(self, u: Cluster, v: Cluster, new_id: int) -> Cluster:
        """
        Merge two clusters (Figure 6 in paper).
        
        1. Combine point indices
        2. Compute new mean
        3. Select c scattered representative points
        4. Shrink representatives toward mean
        """
        # Combine points
        w_points_idx = u.points_idx + v.points_idx
        w_points = self.S[w_points_idx]
        w_mean = np.mean(w_points, axis=0)
        
        # Get candidate representatives by unshrinking from both clusters
        candidates = set()
        for shrunk in u.reps:
            original = self._unshrink_point(shrunk, u.mean)
            candidates.add(tuple(original))
        for shrunk in v.reps:
            original = self._unshrink_point(shrunk, v.mean)
            candidates.add(tuple(original))
        
        candidate_list = [np.array(p) for p in candidates]
        
        # If not enough candidates, use all points in cluster
        if len(candidate_list) < self.c:
            candidate_list = list(w_points)
        
        # Select c scattered points (Figure 6, Steps 4-17)
        selected = []
        
        # First point: farthest from mean
        max_dist = -1
        best_point = None
        for p in candidate_list:
            dist = euclidean_distance(p, w_mean)
            if dist > max_dist:
                max_dist = dist
                best_point = p
        
        if best_point is not None:
            selected.append(best_point)
        
        # Remaining points: maximize minimum distance to already selected
        for _ in range(self.c - 1):
            if not selected:
                break
            
            max_min_dist = -1
            best_point = None
            
            for p in candidate_list:
                # Skip if already selected
                if any(np.array_equal(p, s) for s in selected):
                    continue
                
                # Find minimum distance to selected points
                min_dist = min(euclidean_distance(p, s) for s in selected)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_point = p
            
            if best_point is not None:
                selected.append(best_point)
            else:
                break
        
        # Shrink toward mean (Figure 6, Step 19)
        shrunk_reps = [p + self.alpha * (w_mean - p) for p in selected]
        
        return Cluster(new_id, w_points_idx, shrunk_reps, w_mean)
    
    def fit(self, X: np.ndarray, verbose: bool = False) -> 'CURE':
        """
        Fit the CURE algorithm to data.
        
        Args:
            X: Input data array (n_samples, n_features)
            verbose: Print progress information
            
        Returns:
            self
        """
        self.S = np.asarray(X, dtype=np.float64)
        self.n, self.d = self.S.shape
        
        if self.n < self.k:
            raise ValueError(f"n_samples ({self.n}) must be >= k ({self.k})")
        
        if verbose:
            print(f"CURE: Clustering {self.n} points into {self.k} clusters")
            print(f"  c={self.c} representatives, alpha={self.alpha}")
        
        # Initialize: each point is its own cluster
        clusters: Dict[int, Cluster] = {}
        for i in range(self.n):
            clusters[i] = Cluster(
                cluster_id=i,
                points_idx=[i],
                reps=[self.S[i].copy()],
                mean=self.S[i].copy()
            )
        
        # Find initial closest for each cluster
        if verbose:
            print("  Finding initial nearest neighbors...")
        
        for i in range(self.n):
            clusters[i].closest, clusters[i].dist = self._find_closest_cluster_brute(clusters[i], clusters)
        
        # Build KD-tree (Step 1)
        T, rep_map = self._build_kd_tree(clusters)
        
        # Build heap (Step 2)
        Q = []
        for cluster_id, cluster in clusters.items():
            if cluster.closest is not None:
                heapq.heappush(Q, cluster.to_heap_entry())
        
        next_id = self.n
        
        if verbose:
            print(f"  Starting agglomerative merging...")
        
        # Main loop (Step 3)
        merge_count = 0
        while len(clusters) > self.k:
            if not Q:
                break
            
            # Step 4: Extract minimum
            _, u_id, _ = heapq.heappop(Q)
            
            if u_id not in clusters or not clusters[u_id].alive:
                continue
            
            u = clusters[u_id]
            
            # Step 5: v := u.closest
            v_id = u.closest
            
            if v_id not in clusters or not clusters[v_id].alive:
                # Recompute closest and re-insert
                u.closest, u.dist = self._find_closest_cluster_kdtree(u, T, rep_map)
                if u.closest is None:
                    u.closest, u.dist = self._find_closest_cluster_brute(u, clusters)
                if u.closest is not None:
                    heapq.heappush(Q, u.to_heap_entry())
                continue
            
            v = clusters[v_id]
            
            # Step 7: Merge
            w = self._merge_clusters(u, v, next_id)
            
            # Mark old clusters as dead and remove
            u.alive = False
            v.alive = False
            del clusters[u_id]
            del clusters[v_id]
            clusters[w.id] = w
            
            # Step 8: Update KD-tree
            T, rep_map = self._build_kd_tree(clusters)
            
            # Step 9: Initialize w.closest
            w.dist = float('inf')
            w.closest = None
            
            # Step 10-24: Update closest for all clusters
            for x_id, x in clusters.items():
                if x_id == w.id or not x.alive:
                    continue
                
                needs_relocate = False
                
                # Steps 11-12: Check if w is closer to x than x's current closest
                dist_w_x = cluster_distance(w, x)
                if dist_w_x < w.dist:
                    w.closest = x_id
                    w.dist = dist_w_x
                
                # Steps 13-19: If x had u or v as closest
                if x.closest == u_id or x.closest == v_id:
                    dist_x_w = cluster_distance(x, w)
                    
                    # Step 15: Use KD-tree to find closest within threshold
                    z_id, dist_x_z = self._find_closest_cluster_kdtree(x, T, rep_map, dist_x_w)
                    
                    if z_id is not None and dist_x_z < dist_x_w:
                        x.closest = z_id
                        x.dist = dist_x_z
                    else:
                        x.closest = w.id
                        x.dist = dist_x_w
                    
                    needs_relocate = True
                
                # Steps 20-22: Check if w is closer than current closest
                elif cluster_distance(x, w) < x.dist:
                    x.closest = w.id
                    x.dist = cluster_distance(x, w)
                    needs_relocate = True
                
                # Steps 18, 22: Relocate in heap
                if needs_relocate:
                    heapq.heappush(Q, x.to_heap_entry())
            
            # Step 25: Insert w into heap
            if w.closest is not None:
                heapq.heappush(Q, w.to_heap_entry())
            
            next_id += 1
            merge_count += 1
            
            if verbose and merge_count % 100 == 0:
                print(f"    Merged {merge_count}, {len(clusters)} clusters remaining")
        
        if verbose:
            print(f"  Clustering complete: {len(clusters)} clusters")
        
        # Store results
        self.clusters_ = list(clusters.values())
        
        # Assign labels
        self.labels_ = np.zeros(self.n, dtype=int)
        for label, cluster in enumerate(self.clusters_):
            for idx in cluster.points_idx:
                self.labels_[idx] = label
        
        return self
    
    def fit_predict(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X, verbose)
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data points.
        
        Each point is assigned to the cluster with the nearest representative.
        """
        if not self.clusters_:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i, point in enumerate(X):
            min_dist = float('inf')
            best_label = 0
            
            for label, cluster in enumerate(self.clusters_):
                for rep in cluster.reps:
                    dist = euclidean_distance(point, rep)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
            
            labels[i] = best_label
        
        return labels


# ============================================================
# SCALABLE CURE - FOR LARGE DATASETS
# ============================================================

class Scalable_CURE:
    """
    Scalable CURE Algorithm for Large Datasets.
    
    Uses random sampling and partitioning as described in Section 4 of the paper:
    1. Draw random sample from data
    2. Partition sample into p partitions
    3. Partially cluster each partition (to n/q clusters)
    4. Cluster the partial clusters in second pass
    5. Eliminate outliers
    6. Label all data points using representatives
    
    Attributes:
        k: Number of clusters
        c: Number of representatives per cluster
        alpha: Shrink factor
        sample_size: Fraction or absolute number of samples
        n_partitions: Number of partitions (p)
        reduce_factor: Reduction factor per partition (q)
    """
    
    def __init__(
        self,
        k: int = 5,
        c: int = 5,
        alpha: float = 0.3,
        sample_size: float = 0.1,
        n_partitions: int = 5,
        reduce_factor: int = 3,
        outlier_threshold: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize Scalable CURE.
        
        Args:
            k: Number of clusters
            c: Number of representatives per cluster
            alpha: Shrink factor
            sample_size: Fraction (0-1) or absolute count of samples
            n_partitions: Number of partitions (p)
            reduce_factor: Cluster each partition until 1/q remain (q)
            outlier_threshold: Clusters with <= this many points are outliers
            random_state: Random seed
        """
        self.k = int(k)
        self.c = int(c)
        self.alpha = float(alpha)
        self.sample_size = sample_size
        self.n_partitions = int(n_partitions)
        self.reduce_factor = int(reduce_factor)
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        
        # Set during fit
        self.S: Optional[np.ndarray] = None
        self.n: int = 0
        self.d: int = 0
        self.clusters_: List[Cluster] = []
        self.labels_: Optional[np.ndarray] = None
        self.sample_indices_: Optional[np.ndarray] = None
    
    def _compute_sample_size(self, n: int) -> int:
        """Compute actual sample size."""
        if isinstance(self.sample_size, float) and self.sample_size <= 1.0:
            return max(self.k * 10, int(n * self.sample_size))
        return min(int(self.sample_size), n)
    
    def _partial_cluster(
        self, 
        points: np.ndarray, 
        indices: np.ndarray, 
        target_clusters: int
    ) -> List[Cluster]:
        """
        Partially cluster a partition.
        
        Args:
            points: Data points in partition
            indices: Original indices
            target_clusters: Stop when this many clusters remain
            
        Returns:
            List of partial clusters
        """
        if len(points) <= target_clusters:
            # Return each point as its own cluster
            clusters = []
            for i, (point, idx) in enumerate(zip(points, indices)):
                clusters.append(Cluster(i, [idx], [point.copy()], point.copy()))
            return clusters
        
        # Use base CURE
        cure = CURE(k=target_clusters, c=self.c, alpha=self.alpha)
        cure.fit(points)
        
        # Map labels back to original indices
        result = []
        for label, cluster in enumerate(cure.clusters_):
            original_indices = [indices[i] for i in cluster.points_idx]
            new_cluster = Cluster(
                cluster_id=label,
                points_idx=original_indices,
                reps=[r.copy() for r in cluster.reps],
                mean=cluster.mean.copy()
            )
            result.append(new_cluster)
        
        return result
    
    def _eliminate_outliers(self, clusters: List[Cluster]) -> List[Cluster]:
        """Remove clusters that are likely outliers (too few points)."""
        return [c for c in clusters if len(c.points_idx) > self.outlier_threshold]
    
    def fit(self, X: np.ndarray, verbose: bool = False) -> 'Scalable_CURE':
        """
        Fit Scalable CURE to data.
        
        Args:
            X: Input data (n_samples, n_features)
            verbose: Print progress
            
        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.S = np.asarray(X, dtype=np.float64)
        self.n, self.d = self.S.shape
        
        if verbose:
            print(f"Scalable CURE: {self.n} points -> {self.k} clusters")
        
        # Step 1: Random sampling
        sample_n = self._compute_sample_size(self.n)
        sample_n = min(sample_n, self.n)
        self.sample_indices_ = np.random.choice(self.n, size=sample_n, replace=False)
        sample_data = self.S[self.sample_indices_]
        
        if verbose:
            print(f"  Step 1: Sampled {sample_n} points")
        
        # Step 2: Partition sample
        partition_size = sample_n // self.n_partitions
        shuffled_idx = np.random.permutation(sample_n)
        
        partitions = []
        for i in range(self.n_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.n_partitions - 1 else sample_n
            part_idx = shuffled_idx[start:end]
            partitions.append((
                sample_data[part_idx],
                self.sample_indices_[part_idx]
            ))
        
        if verbose:
            print(f"  Step 2: Created {self.n_partitions} partitions")
        
        # Step 3: Partial clustering on each partition
        target_per_partition = max(self.k, partition_size // self.reduce_factor)
        
        all_clusters = []
        for i, (part_data, part_indices) in enumerate(partitions):
            if len(part_data) == 0:
                continue
            
            target = min(target_per_partition, len(part_data))
            partial_clusters = self._partial_cluster(part_data, part_indices, target)
            
            # Remove small clusters (outliers) from partition
            partial_clusters = [c for c in partial_clusters if len(c.points_idx) > 1]
            all_clusters.extend(partial_clusters)
            
            if verbose:
                print(f"    Partition {i+1}: {len(part_data)} pts -> {len(partial_clusters)} clusters")
        
        if verbose:
            print(f"  Step 3: Total partial clusters: {len(all_clusters)}")
        
        # Step 4: Second pass - cluster the partial clusters
        if len(all_clusters) > self.k:
            # Build a synthetic dataset from cluster representatives
            # and run CURE on it
            rep_data = []
            rep_to_cluster = []
            
            for i, cluster in enumerate(all_clusters):
                for rep in cluster.reps:
                    rep_data.append(rep)
                    rep_to_cluster.append(i)
            
            rep_data = np.array(rep_data)
            
            # Run CURE on representatives
            cure = CURE(k=self.k, c=self.c, alpha=self.alpha)
            cure.fit(rep_data)
            
            # Map back to original clusters
            final_clusters = []
            for label, cure_cluster in enumerate(cure.clusters_):
                merged_points = []
                merged_reps = []
                
                for rep_idx in cure_cluster.points_idx:
                    original_cluster = all_clusters[rep_to_cluster[rep_idx]]
                    merged_points.extend(original_cluster.points_idx)
                
                # Use the CURE cluster's representatives
                merged_reps = [r.copy() for r in cure_cluster.reps]
                
                final_clusters.append(Cluster(
                    cluster_id=label,
                    points_idx=list(set(merged_points)),
                    reps=merged_reps,
                    mean=cure_cluster.mean.copy()
                ))
            
            all_clusters = final_clusters
        
        if verbose:
            print(f"  Step 4: Merged to {len(all_clusters)} clusters")
        
        # Step 5: Eliminate outliers
        self.clusters_ = self._eliminate_outliers(all_clusters)
        
        if verbose:
            print(f"  Step 5: After outlier removal: {len(self.clusters_)} clusters")
        
        # Step 6: Label all data points
        self.labels_ = self._assign_labels()
        
        if verbose:
            print(f"  Step 6: Labeled all {self.n} points")
        
        return self
    
    def _assign_labels(self) -> np.ndarray:
        """Assign labels to all data points using nearest representative."""
        labels = np.zeros(self.n, dtype=int)
        
        # Collect all representatives
        all_reps = []
        rep_labels = []
        for label, cluster in enumerate(self.clusters_):
            for rep in cluster.reps:
                all_reps.append(rep)
                rep_labels.append(label)
        
        if not all_reps:
            return labels
        
        all_reps = np.array(all_reps)
        rep_labels = np.array(rep_labels)
        
        # Build KD-tree for fast assignment
        tree = cKDTree(all_reps)
        
        # Assign each point to nearest representative's cluster
        _, indices = tree.query(self.S)
        labels = rep_labels[indices]
        
        return labels
    
    def fit_predict(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Fit and return labels."""
        self.fit(X, verbose)
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if not self.clusters_:
            raise ValueError("Model not fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        
        all_reps = []
        rep_labels = []
        for label, cluster in enumerate(self.clusters_):
            for rep in cluster.reps:
                all_reps.append(rep)
                rep_labels.append(label)
        
        tree = cKDTree(np.array(all_reps))
        _, indices = tree.query(X)
        
        return np.array(rep_labels)[indices]


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def cure_clustering(
    X: np.ndarray,
    k: int = 5,
    c: int = 5,
    alpha: float = 0.3,
    scalable: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for CURE clustering.
    
    Args:
        X: Data array (n_samples, n_features)
        k: Number of clusters
        c: Number of representatives per cluster
        alpha: Shrink factor
        scalable: Use Scalable_CURE for large datasets
        **kwargs: Additional arguments for Scalable_CURE
        
    Returns:
        Cluster labels
    """
    if scalable or len(X) > 5000:
        model = Scalable_CURE(k=k, c=c, alpha=alpha, **kwargs)
    else:
        model = CURE(k=k, c=c, alpha=alpha)
    
    return model.fit_predict(X)


# ============================================================
# MAIN - TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Euclidean CURE...")
    
    # Generate test data
    np.random.seed(42)
    
    # Create 3 clusters
    c1 = np.random.randn(100, 2) * 2 + [0, 0]
    c2 = np.random.randn(100, 2) * 2 + [10, 10]
    c3 = np.random.randn(100, 2) * 2 + [10, 0]
    
    X = np.vstack([c1, c2, c3])
    true_labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    # Test CURE
    print("\nBase CURE:")
    cure = CURE(k=3, c=5, alpha=0.3)
    labels = cure.fit_predict(X, verbose=True)
    
    # Metrics
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(true_labels, labels)
    print(f"ARI: {ari:.4f}")
    
    # Test Scalable CURE
    print("\nScalable CURE:")
    scalable = Scalable_CURE(k=3, c=5, alpha=0.3, sample_size=0.5)
    labels2 = scalable.fit_predict(X, verbose=True)
    ari2 = adjusted_rand_score(true_labels, labels2)
    print(f"ARI: {ari2:.4f}")
