"""
CURE Algorithm - Pearson Correlation Distance Version

Implementation based on the CURE paper, adapted for Pearson correlation distance.
Pearson distance captures similarity in patterns/trends rather than absolute positions.

Contains:
- CURE: Base hierarchical clustering algorithm using Pearson distance
- Scalable_CURE: Optimized version for large datasets

Note: Pearson correlation works best with higher-dimensional data (>3 features).
For low-dimensional spatial data, use Euclidean distance instead.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Any
import warnings

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore


# ============================================================
# DISTANCE FUNCTIONS
# ============================================================

def pearson_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Pearson correlation distance: d = 1 - correlation(p, q)
    
    Returns value in [0, 2]:
        0 = identical patterns (correlation = 1)
        1 = no correlation (correlation = 0)
        2 = opposite patterns (correlation = -1)
    
    Args:
        p, q: Feature vectors
        
    Returns:
        Pearson distance
    """
    if len(p) < 2 or len(q) < 2:
        # Fallback for very low dimensional data
        diff = np.linalg.norm(p - q)
        max_norm = max(np.linalg.norm(p), np.linalg.norm(q), 1e-10)
        return min(diff / max_norm, 2.0)
    
    # Center the vectors
    p_centered = p - np.mean(p)
    q_centered = q - np.mean(q)
    
    # Compute norms
    p_norm = np.linalg.norm(p_centered)
    q_norm = np.linalg.norm(q_centered)
    
    # Handle zero variance
    if p_norm < 1e-10 or q_norm < 1e-10:
        return 1.0  # Undefined correlation -> neutral distance
    
    # Pearson correlation
    correlation = np.dot(p_centered, q_centered) / (p_norm * q_norm)
    
    # Clamp for numerical stability
    correlation = np.clip(correlation, -1.0, 1.0)
    
    return 1.0 - correlation


def cluster_distance(u: 'Cluster', v: 'Cluster') -> float:
    """
    Compute distance between two clusters using Pearson correlation.
    
    Distance = minimum Pearson distance between any pair of representatives.
    Uses vectorized computation for efficiency.
    """
    if len(u.reps) == 0 or len(v.reps) == 0:
        return float('inf')
    
    u_reps = np.array(u.reps)
    v_reps = np.array(v.reps)
    
    # Handle single-feature case
    if u_reps.shape[1] < 2:
        # Fall back to normalized Euclidean
        min_dist = float('inf')
        for p in u_reps:
            for q in v_reps:
                dist = np.abs(p[0] - q[0]) / max(abs(p[0]), abs(q[0]), 1e-10)
                min_dist = min(min_dist, dist)
        return min_dist
    
    # Center and normalize each representative
    u_centered = u_reps - np.mean(u_reps, axis=1, keepdims=True)
    v_centered = v_reps - np.mean(v_reps, axis=1, keepdims=True)
    
    u_norms = np.linalg.norm(u_centered, axis=1, keepdims=True)
    v_norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
    
    u_norms = np.where(u_norms < 1e-10, 1.0, u_norms)
    v_norms = np.where(v_norms < 1e-10, 1.0, v_norms)
    
    u_normalized = u_centered / u_norms
    v_normalized = v_centered / v_norms
    
    # Correlation matrix: (n_u, n_v)
    corr_matrix = np.dot(u_normalized, v_normalized.T)
    
    # Distance = 1 - correlation, we want min distance = 1 - max correlation
    return 1.0 - np.max(corr_matrix)


# ============================================================
# CLUSTER CLASS
# ============================================================

class Cluster:
    """
    Represents a cluster in the CURE algorithm.
    
    For Pearson correlation, the centroid is computed as the medoid
    (actual point that minimizes total distance to other points)
    rather than the arithmetic mean.
    """
    
    def __init__(self, cluster_id: int, points_idx: List[int], reps: List[np.ndarray], 
                 mean: np.ndarray, points: Optional[np.ndarray] = None):
        self.id = cluster_id
        self.points_idx = points_idx if isinstance(points_idx, list) else [points_idx]
        self.reps = reps if isinstance(reps, list) else [reps]
        self.mean = mean  # For Pearson, this is the medoid
        self.points = points  # Store actual points for medoid computation
        self.alive = True
        self.dist = float('inf')
        self.closest: Optional[int] = None
    
    def __lt__(self, other):
        return self.dist < other.dist
    
    def to_heap_entry(self) -> Tuple[float, int, 'Cluster']:
        return (self.dist, self.id, self)
    
    def compute_medoid(self) -> np.ndarray:
        """
        Compute medoid - the point that minimizes total Pearson distance.
        """
        if self.points is None or len(self.points) <= 1:
            return self.mean.copy()
        
        n = len(self.points)
        
        if n <= 50:
            # Brute force for small clusters
            min_total_dist = float('inf')
            medoid = self.points[0].copy()
            
            for i, p in enumerate(self.points):
                total_dist = sum(pearson_distance(p, q) for j, q in enumerate(self.points) if i != j)
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    medoid = p.copy()
            
            return medoid
        else:
            # Vectorized for larger clusters
            centered = self.points - np.mean(self.points, axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            normalized = centered / norms
            
            # Correlation matrix
            corr_matrix = np.dot(normalized, normalized.T)
            dist_matrix = 1.0 - corr_matrix
            
            # Sum of distances for each point
            total_dists = np.sum(dist_matrix, axis=1)
            medoid_idx = np.argmin(total_dists)
            
            return self.points[medoid_idx].copy()


# ============================================================
# CURE CLASS - PEARSON VERSION
# ============================================================

class CURE:
    """
    CURE Algorithm using Pearson Correlation Distance.
    
    Attributes:
        k: Number of clusters
        c: Number of representative points per cluster
        alpha: Shrink factor
        standardize: Whether to standardize features before clustering
    """
    
    def __init__(self, k: int = 5, c: int = 5, alpha: float = 0.3, standardize: bool = True):
        """
        Initialize CURE.
        
        Args:
            k: Number of clusters
            c: Number of representatives per cluster
            alpha: Shrink factor (0 to 1)
            standardize: Standardize features (recommended for Pearson)
        """
        self.k = int(k)
        self.c = int(c)
        self.alpha = float(alpha)
        self.standardize = standardize
        
        # Set during fit
        self.S: Optional[np.ndarray] = None
        self.n: int = 0
        self.d: int = 0
        self.clusters_: List[Cluster] = []
        self.labels_: Optional[np.ndarray] = None
        self.representatives_: List[np.ndarray] = []
        
        # Standardization parameters
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
    
    def _standardize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Standardize features to zero mean, unit variance."""
        if fit:
            self._feature_mean = np.mean(X, axis=0)
            self._feature_std = np.std(X, axis=0)
            self._feature_std = np.where(self._feature_std < 1e-10, 1.0, self._feature_std)
        
        return (X - self._feature_mean) / self._feature_std
    
    def _destandardize(self, X: np.ndarray) -> np.ndarray:
        """Convert standardized data back to original scale."""
        if self._feature_mean is None:
            return X
        return X * self._feature_std + self._feature_mean
    
    def _find_closest_cluster(
        self, 
        query: Cluster, 
        clusters: Dict[int, Cluster]
    ) -> Tuple[Optional[int], float]:
        """Find closest cluster to query (full scan with Pearson distance)."""
        min_dist = float('inf')
        closest_id = None
        
        for cluster_id, cluster in clusters.items():
            if cluster_id == query.id or not cluster.alive:
                continue
            
            dist = cluster_distance(query, cluster)
            if dist < min_dist:
                min_dist = dist
                closest_id = cluster_id
        
        return closest_id, min_dist

    def _build_rep_tree(
        self, clusters: Dict[int, Cluster]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
        """Build (rep_points, rep_map, cKDTree) from current clusters for Euclidean NN.
        rep_map[i] = cluster_id for rep_points[i]. Returns (rep_points, rep_map, tree).
        tree is None if scipy not available."""
        rep_points, rep_map = [], []
        for cid, cluster in clusters.items():
            if not cluster.alive:
                continue
            for rep in cluster.reps:
                rep_points.append(rep)
                rep_map.append(cid)
        if not rep_points:
            return np.array([]), np.array([], dtype=int), None
        rep_points = np.array(rep_points)
        rep_map = np.array(rep_map)
        tree = cKDTree(rep_points) if cKDTree is not None else None
        return rep_points, rep_map, tree

    def _find_closest_cluster_with_tree(
        self,
        query: Cluster,
        clusters: Dict[int, Cluster],
        rep_points: np.ndarray,
        rep_map: np.ndarray,
        tree: Optional[Any],
    ) -> Tuple[Optional[int], float]:
        """Find closest cluster using Euclidean KD-tree candidates, then Pearson for tie-break.
        If tree is None, falls back to full _find_closest_cluster."""
        if tree is None or rep_points.size == 0:
            return self._find_closest_cluster(query, clusters)
        k = min(len(rep_points), self.c * 2 + 2)
        candidate_ids = set()
        for rep in query.reps:
            rep_ = np.asarray(rep).reshape(1, -1)
            _, indices = tree.query(rep_, k=k)
            indices = np.atleast_1d(indices).ravel()
            for idx in indices:
                idx = int(idx)
                if 0 <= idx < len(rep_map):
                    cid = int(rep_map[idx])
                    if cid != query.id and cid in clusters and clusters[cid].alive:
                        candidate_ids.add(cid)
        if not candidate_ids:
            return self._find_closest_cluster(query, clusters)
        min_dist = float('inf')
        closest_id = None
        for cid in candidate_ids:
            dist = cluster_distance(query, clusters[cid])
            if dist < min_dist:
                min_dist = dist
                closest_id = cid
        return closest_id, min_dist

    def _unshrink_point(self, shrunk: np.ndarray, medoid: np.ndarray) -> np.ndarray:
        """Recover original point from shrunk representative."""
        if self.alpha == 1.0:
            return medoid.copy()
        return (shrunk - self.alpha * medoid) / (1.0 - self.alpha)
    
    def _merge_clusters(self, u: Cluster, v: Cluster, new_id: int) -> Cluster:
        """Merge two clusters."""
        # Combine points
        w_points_idx = u.points_idx + v.points_idx
        w_points = self.S[w_points_idx]
        
        # Create temporary cluster to compute medoid
        temp_cluster = Cluster(new_id, w_points_idx, [], np.mean(w_points, axis=0), w_points)
        w_medoid = temp_cluster.compute_medoid()
        
        # Get candidate representatives by unshrinking
        candidates = set()
        for shrunk in u.reps:
            original = self._unshrink_point(shrunk, u.mean)
            candidates.add(tuple(original))
        for shrunk in v.reps:
            original = self._unshrink_point(shrunk, v.mean)
            candidates.add(tuple(original))
        
        candidate_list = [np.array(p) for p in candidates]
        
        if len(candidate_list) < self.c:
            candidate_list = list(w_points)
        
        # Select scattered representatives
        selected = []
        
        # First point: farthest from medoid
        max_dist = -1
        best_point = None
        for p in candidate_list:
            dist = pearson_distance(p, w_medoid)
            if dist > max_dist:
                max_dist = dist
                best_point = p
        
        if best_point is not None:
            selected.append(best_point)
        
        # Remaining points
        for _ in range(self.c - 1):
            if not selected:
                break
            
            max_min_dist = -1
            best_point = None
            
            for p in candidate_list:
                if any(np.array_equal(p, s) for s in selected):
                    continue
                
                min_dist = min(pearson_distance(p, s) for s in selected)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_point = p
            
            if best_point is not None:
                selected.append(best_point)
            else:
                break
        
        # Shrink toward medoid
        shrunk_reps = [p + self.alpha * (w_medoid - p) for p in selected]
        
        return Cluster(new_id, w_points_idx, shrunk_reps, w_medoid, w_points)
    
    def fit(self, X: np.ndarray, verbose: bool = False) -> 'CURE':
        """
        Fit CURE to data.
        
        Args:
            X: Input data (n_samples, n_features)
            verbose: Print progress
            
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        
        if self.standardize:
            self.S = self._standardize_data(X, fit=True)
        else:
            self.S = X.copy()
        
        self.n, self.d = self.S.shape
        
        if self.n < self.k:
            raise ValueError(f"n_samples ({self.n}) must be >= k ({self.k})")
        
        if self.d <= 3:
            warnings.warn(
                f"Pearson correlation with {self.d} features may not capture spatial patterns well. "
                "Consider using Euclidean distance for low-dimensional data.",
                UserWarning
            )
        
        if verbose:
            print(f"Pearson CURE: Clustering {self.n} points into {self.k} clusters")
            print(f"  c={self.c} representatives, alpha={self.alpha}")
        
        # Initialize clusters
        clusters: Dict[int, Cluster] = {}
        for i in range(self.n):
            clusters[i] = Cluster(
                cluster_id=i,
                points_idx=[i],
                reps=[self.S[i].copy()],
                mean=self.S[i].copy(),
                points=self.S[i:i+1].copy()
            )
        
        # Find initial closest (use Euclidean KD-tree for candidate search to avoid O(n^2) Pearson scans)
        if verbose:
            print("  Finding initial nearest neighbors...")
        rep_points, rep_map, tree = self._build_rep_tree(clusters)
        for i in range(self.n):
            clusters[i].closest, clusters[i].dist = self._find_closest_cluster_with_tree(
                clusters[i], clusters, rep_points, rep_map, tree
            )
        
        # Build heap
        Q = []
        for cluster in clusters.values():
            if cluster.closest is not None:
                heapq.heappush(Q, cluster.to_heap_entry())
        
        next_id = self.n
        
        if verbose:
            print(f"  Starting agglomerative merging...")
        
        merge_count = 0
        while len(clusters) > self.k:
            if not Q:
                break
            
            _, u_id, _ = heapq.heappop(Q)
            
            if u_id not in clusters or not clusters[u_id].alive:
                continue
            
            u = clusters[u_id]
            v_id = u.closest
            
            if v_id not in clusters or not clusters[v_id].alive:
                _rp, _rm, _t = self._build_rep_tree(clusters)
                u.closest, u.dist = self._find_closest_cluster_with_tree(u, clusters, _rp, _rm, _t)
                if u.closest is not None:
                    heapq.heappush(Q, u.to_heap_entry())
                continue
            
            v = clusters[v_id]
            
            # Merge
            w = self._merge_clusters(u, v, next_id)
            
            u.alive = False
            v.alive = False
            del clusters[u_id]
            del clusters[v_id]
            clusters[w.id] = w
            
            # Initialize w.closest
            w.dist = float('inf')
            w.closest = None

            # Build rep tree once per merge for fast closest-cluster search
            rep_points, rep_map, tree = self._build_rep_tree(clusters)
            
            # Update all clusters
            for x_id, x in clusters.items():
                if x_id == w.id or not x.alive:
                    continue
                
                needs_relocate = False
                
                # Check if w is closer
                dist_w_x = cluster_distance(w, x)
                if dist_w_x < w.dist:
                    w.closest = x_id
                    w.dist = dist_w_x
                
                # If x had u or v as closest
                if x.closest == u_id or x.closest == v_id:
                    dist_x_w = cluster_distance(x, w)
                    
                    # Find new closest (KD-tree candidates, then Pearson)
                    best_id, best_dist = self._find_closest_cluster_with_tree(
                        x, clusters, rep_points, rep_map, tree
                    )
                    
                    if best_dist < dist_x_w:
                        x.closest = best_id
                        x.dist = best_dist
                    else:
                        x.closest = w.id
                        x.dist = dist_x_w
                    
                    needs_relocate = True
                
                # Check if w is better than current closest
                elif cluster_distance(x, w) < x.dist:
                    x.closest = w.id
                    x.dist = cluster_distance(x, w)
                    needs_relocate = True
                
                if needs_relocate:
                    heapq.heappush(Q, x.to_heap_entry())
            
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
        
        # Store representatives (de-standardize if needed)
        if self.standardize and self._feature_mean is not None:
            self.representatives_ = [
                [self._destandardize(r) for r in c.reps] if c.reps else []
                for c in self.clusters_
            ]
        else:
            self.representatives_ = [c.reps for c in self.clusters_]
        
        # Assign labels
        self.labels_ = np.zeros(self.n, dtype=int)
        for label, cluster in enumerate(self.clusters_):
            for idx in cluster.points_idx:
                self.labels_[idx] = label
        
        return self
    
    def fit_predict(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Fit and return labels."""
        self.fit(X, verbose)
        return self.labels_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if not self.clusters_:
            raise ValueError("Model not fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.standardize:
            X = self._standardize_data(X, fit=False)
        
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i, point in enumerate(X):
            min_dist = float('inf')
            best_label = 0
            
            for label, cluster in enumerate(self.clusters_):
                for rep in cluster.reps:
                    dist = pearson_distance(point, rep)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
            
            labels[i] = best_label
        
        return labels


# ============================================================
# SCALABLE CURE - PEARSON VERSION
# ============================================================

class Scalable_CURE:
    """
    Scalable CURE Algorithm using Pearson Correlation Distance.
    
    Uses sampling and partitioning for large datasets.
    """
    
    def __init__(
        self,
        k: int = 5,
        c: int = 5,
        alpha: float = 0.3,
        standardize: bool = True,
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
            c: Number of representatives
            alpha: Shrink factor
            standardize: Standardize features
            sample_size: Fraction or count of samples
            n_partitions: Number of partitions
            reduce_factor: Reduction per partition
            outlier_threshold: Min points for non-outlier cluster
            random_state: Random seed
        """
        self.k = int(k)
        self.c = int(c)
        self.alpha = float(alpha)
        self.standardize = standardize
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
        self.representatives_: List[np.ndarray] = []
        
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
    
    def _standardize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Standardize features."""
        if fit:
            self._feature_mean = np.mean(X, axis=0)
            self._feature_std = np.std(X, axis=0)
            self._feature_std = np.where(self._feature_std < 1e-10, 1.0, self._feature_std)
        return (X - self._feature_mean) / self._feature_std
    
    def _destandardize(self, X: np.ndarray) -> np.ndarray:
        """Convert back to original scale."""
        if self._feature_mean is None:
            return X
        return X * self._feature_std + self._feature_mean
    
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
        """Partially cluster a partition."""
        if len(points) <= target_clusters:
            clusters = []
            for i, (point, idx) in enumerate(zip(points, indices)):
                clusters.append(Cluster(i, [idx], [point.copy()], point.copy(), point.reshape(1, -1)))
            return clusters
        
        cure = CURE(k=target_clusters, c=self.c, alpha=self.alpha, standardize=False)
        cure.S = points
        cure.n = len(points)
        cure.d = points.shape[1]
        
        # Initialize
        clusters_dict = {}
        for i in range(len(points)):
            clusters_dict[i] = Cluster(i, [i], [points[i].copy()], points[i].copy(), points[i:i+1])
        
        for i in range(len(points)):
            clusters_dict[i].closest, clusters_dict[i].dist = cure._find_closest_cluster(clusters_dict[i], clusters_dict)
        
        Q = []
        for c in clusters_dict.values():
            if c.closest is not None:
                heapq.heappush(Q, c.to_heap_entry())
        
        next_id = len(points)
        
        while len(clusters_dict) > target_clusters:
            if not Q:
                break
            
            _, u_id, _ = heapq.heappop(Q)
            
            if u_id not in clusters_dict or not clusters_dict[u_id].alive:
                continue
            
            u = clusters_dict[u_id]
            v_id = u.closest
            
            if v_id not in clusters_dict or not clusters_dict[v_id].alive:
                u.closest, u.dist = cure._find_closest_cluster(u, clusters_dict)
                if u.closest is not None:
                    heapq.heappush(Q, u.to_heap_entry())
                continue
            
            v = clusters_dict[v_id]
            
            # Merge
            w_points_idx = u.points_idx + v.points_idx
            w_points = points[w_points_idx]
            w = Cluster(next_id, w_points_idx, [], np.mean(w_points, axis=0), w_points)
            w.mean = w.compute_medoid()
            
            # Get representatives
            candidates = list(set([tuple(r) for r in u.reps] + [tuple(r) for r in v.reps]))
            candidates = [np.array(c) for c in candidates]
            if len(candidates) < self.c:
                candidates = list(w_points)
            
            selected = []
            if candidates:
                # Select scattered representatives
                max_dist = -1
                best = None
                for p in candidates:
                    dist = pearson_distance(p, w.mean)
                    if dist > max_dist:
                        max_dist = dist
                        best = p
                if best is not None:
                    selected.append(best)
                
                for _ in range(self.c - 1):
                    if not selected:
                        break
                    max_min = -1
                    best = None
                    for p in candidates:
                        if any(np.array_equal(p, s) for s in selected):
                            continue
                        min_d = min(pearson_distance(p, s) for s in selected)
                        if min_d > max_min:
                            max_min = min_d
                            best = p
                    if best is not None:
                        selected.append(best)
                    else:
                        break
            
            # Shrink
            w.reps = [p + self.alpha * (w.mean - p) for p in selected]
            
            u.alive = False
            v.alive = False
            del clusters_dict[u_id]
            del clusters_dict[v_id]
            clusters_dict[w.id] = w
            
            w.dist = float('inf')
            w.closest = None
            
            for x_id, x in clusters_dict.items():
                if x_id == w.id or not x.alive:
                    continue
                
                dist_w_x = cluster_distance(w, x)
                if dist_w_x < w.dist:
                    w.closest = x_id
                    w.dist = dist_w_x
                
                if x.closest == u_id or x.closest == v_id:
                    best_id, best_dist = cure._find_closest_cluster(x, clusters_dict)
                    x.closest = best_id
                    x.dist = best_dist
                    heapq.heappush(Q, x.to_heap_entry())
                elif cluster_distance(x, w) < x.dist:
                    x.closest = w.id
                    x.dist = cluster_distance(x, w)
                    heapq.heappush(Q, x.to_heap_entry())
            
            if w.closest is not None:
                heapq.heappush(Q, w.to_heap_entry())
            
            next_id += 1
        
        # Map back to original indices
        result = []
        for cluster in clusters_dict.values():
            original_indices = [indices[i] for i in cluster.points_idx]
            new_cluster = Cluster(
                cluster_id=cluster.id,
                points_idx=original_indices,
                reps=[r.copy() for r in cluster.reps],
                mean=cluster.mean.copy(),
                points=points[cluster.points_idx] if cluster.points_idx else None
            )
            result.append(new_cluster)
        
        return result
    
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
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.standardize:
            self.S = self._standardize_data(X, fit=True)
        else:
            self.S = X.copy()
        
        self.n, self.d = self.S.shape
        
        if self.d <= 3:
            warnings.warn(
                f"Pearson correlation with {self.d} features may not capture spatial patterns well.",
                UserWarning
            )
        
        if verbose:
            print(f"Scalable Pearson CURE: {self.n} points -> {self.k} clusters")
        
        # Step 1: Random sampling
        sample_n = self._compute_sample_size(self.n)
        sample_n = min(sample_n, self.n)
        sample_indices = np.random.choice(self.n, size=sample_n, replace=False)
        sample_data = self.S[sample_indices]
        
        if verbose:
            print(f"  Step 1: Sampled {sample_n} points")
        
        # Step 2: Partition
        partition_size = sample_n // self.n_partitions
        shuffled_idx = np.random.permutation(sample_n)
        
        partitions = []
        for i in range(self.n_partitions):
            start = i * partition_size
            end = start + partition_size if i < self.n_partitions - 1 else sample_n
            part_idx = shuffled_idx[start:end]
            partitions.append((sample_data[part_idx], sample_indices[part_idx]))
        
        if verbose:
            print(f"  Step 2: Created {self.n_partitions} partitions")
        
        # Step 3: Partial clustering
        target_per_partition = max(self.k, partition_size // self.reduce_factor)
        
        all_clusters = []
        for i, (part_data, part_indices) in enumerate(partitions):
            if len(part_data) == 0:
                continue
            
            target = min(target_per_partition, len(part_data))
            partial = self._partial_cluster(part_data, part_indices, target)
            partial = [c for c in partial if len(c.points_idx) > 1]
            all_clusters.extend(partial)
            
            if verbose:
                print(f"    Partition {i+1}: {len(part_data)} pts -> {len(partial)} clusters")
        
        if verbose:
            print(f"  Step 3: Total partial clusters: {len(all_clusters)}")
        
        # Step 4: Second pass
        if len(all_clusters) > self.k:
            rep_data = []
            rep_to_cluster = []
            
            for i, cluster in enumerate(all_clusters):
                for rep in cluster.reps:
                    rep_data.append(rep)
                    rep_to_cluster.append(i)
            
            rep_data = np.array(rep_data)
            
            cure = CURE(k=self.k, c=self.c, alpha=self.alpha, standardize=False)
            cure.S = rep_data
            cure.n = len(rep_data)
            cure.d = rep_data.shape[1]
            cure.fit(rep_data)
            
            final_clusters = []
            for label, cure_cluster in enumerate(cure.clusters_):
                merged_points = []
                
                for rep_idx in cure_cluster.points_idx:
                    original_cluster = all_clusters[rep_to_cluster[rep_idx]]
                    merged_points.extend(original_cluster.points_idx)
                
                final_clusters.append(Cluster(
                    cluster_id=label,
                    points_idx=list(set(merged_points)),
                    reps=[r.copy() for r in cure_cluster.reps],
                    mean=cure_cluster.mean.copy()
                ))
            
            all_clusters = final_clusters
        
        if verbose:
            print(f"  Step 4: Merged to {len(all_clusters)} clusters")
        
        # Step 5: Remove outliers
        self.clusters_ = [c for c in all_clusters if len(c.points_idx) > self.outlier_threshold]
        
        if verbose:
            print(f"  Step 5: After outlier removal: {len(self.clusters_)} clusters")
        
        # Store representatives (de-standardize)
        if self.standardize and self._feature_mean is not None:
            self.representatives_ = [
                [self._destandardize(r) for r in c.reps] if c.reps else []
                for c in self.clusters_
            ]
        else:
            self.representatives_ = [c.reps for c in self.clusters_]
        
        # Step 6: Label all points
        self.labels_ = self._assign_labels()
        
        if verbose:
            print(f"  Step 6: Labeled all {self.n} points")
        
        return self
    
    def _assign_labels(self) -> np.ndarray:
        """Assign labels using nearest representative."""
        labels = np.zeros(self.n, dtype=int)
        
        for i, point in enumerate(self.S):
            min_dist = float('inf')
            best_label = 0
            
            for label, cluster in enumerate(self.clusters_):
                for rep in cluster.reps:
                    dist = pearson_distance(point, rep)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
            
            labels[i] = best_label
        
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
        
        if self.standardize:
            X = self._standardize_data(X, fit=False)
        
        labels = np.zeros(X.shape[0], dtype=int)
        
        for i, point in enumerate(X):
            min_dist = float('inf')
            best_label = 0
            
            for label, cluster in enumerate(self.clusters_):
                for rep in cluster.reps:
                    dist = pearson_distance(point, rep)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
            
            labels[i] = best_label
        
        return labels


# ============================================================
# MAIN - TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Pearson CURE...")
    
    # Generate higher-dimensional test data for Pearson
    np.random.seed(42)
    
    # Create 3 clusters with correlated patterns (10 features)
    n_features = 10
    n_per_cluster = 100
    
    # Base patterns
    pattern1 = np.sin(np.linspace(0, 2*np.pi, n_features))
    pattern2 = np.cos(np.linspace(0, 2*np.pi, n_features))
    pattern3 = np.linspace(0, 1, n_features)
    
    # Generate clusters with noise
    c1 = pattern1 + np.random.randn(n_per_cluster, n_features) * 0.2
    c2 = pattern2 + np.random.randn(n_per_cluster, n_features) * 0.2
    c3 = pattern3 + np.random.randn(n_per_cluster, n_features) * 0.2
    
    X = np.vstack([c1, c2, c3])
    true_labels = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)
    
    # Test CURE
    print("\nBase Pearson CURE:")
    cure = CURE(k=3, c=5, alpha=0.3, standardize=True)
    labels = cure.fit_predict(X, verbose=True)
    
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(true_labels, labels)
    print(f"ARI: {ari:.4f}")
    
    # Test Scalable CURE  
    print("\nScalable Pearson CURE:")
    scalable = Scalable_CURE(k=3, c=5, alpha=0.3, standardize=True, sample_size=0.5)
    labels2 = scalable.fit_predict(X, verbose=True)
    ari2 = adjusted_rand_score(true_labels, labels2)
    print(f"ARI: {ari2:.4f}")
