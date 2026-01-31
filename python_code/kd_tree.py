"""
KD-Tree Implementation for CURE Algorithm

This module provides:
1. SelfKDTree: Self-implemented KD-tree from scratch
2. WrapperKDTree: Wrapper using scipy's cKDTree
3. RepresentativeTree: Tree for cluster representatives

Based on the CURE paper: "CURE: An Efficient Clustering Algorithm for Large Databases"
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from scipy.spatial import cKDTree


# ============================================================
# SELF-IMPLEMENTED KD-TREE
# ============================================================

class KDNode:
    """A node in the KD-tree."""
    
    def __init__(self, point: np.ndarray, index: int, axis: int):
        self.point = point
        self.index = index  # Original index in the data array
        self.axis = axis    # Splitting axis
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None


class SelfKDTree:
    """
    Self-implemented KD-tree for nearest neighbor search.
    
    A KD-tree is a binary tree where each node represents a k-dimensional point.
    The tree is built by recursively splitting the data along different axes.
    
    Time complexity:
    - Build: O(n log n)
    - Query (average): O(log n)
    - Query (worst): O(n) for highly unbalanced trees
    """
    
    def __init__(self, data: np.ndarray):
        """
        Build KD-tree from data points.
        
        Args:
            data: Array of shape (n_points, n_dimensions)
        """
        self.data = np.asarray(data, dtype=np.float64)
        self.n_points, self.n_dims = self.data.shape
        
        # Build tree
        indices = list(range(self.n_points))
        self.root = self._build(indices, depth=0)
    
    def _build(self, indices: List[int], depth: int) -> Optional[KDNode]:
        """Recursively build the KD-tree."""
        if not indices:
            return None
        
        # Choose axis based on depth (cycle through dimensions)
        axis = depth % self.n_dims
        
        # Sort indices by the value along current axis
        indices.sort(key=lambda i: self.data[i, axis])
        
        # Choose median as pivot
        mid = len(indices) // 2
        
        # Create node
        node = KDNode(
            point=self.data[indices[mid]],
            index=indices[mid],
            axis=axis
        )
        
        # Recursively build left and right subtrees
        node.left = self._build(indices[:mid], depth + 1)
        node.right = self._build(indices[mid + 1:], depth + 1)
        
        return node
    
    def query(self, point: np.ndarray, k: int = 1, 
              distance_upper_bound: float = float('inf')) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to query point.
        
        Args:
            point: Query point
            k: Number of neighbors to find
            distance_upper_bound: Maximum distance to consider
            
        Returns:
            (distances, indices) arrays of length k
        """
        point = np.asarray(point, dtype=np.float64)
        
        # Use a list to store k best candidates: (distance, index)
        best = []
        
        def _search(node: Optional[KDNode], depth: int):
            if node is None:
                return
            
            # Compute distance to current node
            dist = np.linalg.norm(point - node.point)
            
            # Add to best if within bounds and better than worst in best
            if dist < distance_upper_bound:
                if len(best) < k:
                    best.append((dist, node.index))
                    best.sort(key=lambda x: x[0])
                elif dist < best[-1][0]:
                    best[-1] = (dist, node.index)
                    best.sort(key=lambda x: x[0])
            
            # Determine which subtree to search first
            axis = node.axis
            diff = point[axis] - node.point[axis]
            
            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
            
            # Search the closer subtree first
            _search(first, depth + 1)
            
            # Check if we need to search the other subtree
            # Only if the splitting plane is closer than our worst candidate
            worst_dist = best[-1][0] if len(best) == k else distance_upper_bound
            if abs(diff) < worst_dist:
                _search(second, depth + 1)
        
        _search(self.root, 0)
        
        # Pad results if fewer than k neighbors found
        while len(best) < k:
            best.append((float('inf'), self.n_points))  # Invalid index
        
        distances = np.array([b[0] for b in best])
        indices = np.array([b[1] for b in best])
        
        return distances, indices
    
    def query_ball_point(self, point: np.ndarray, r: float) -> List[int]:
        """
        Find all points within radius r of query point.
        
        Args:
            point: Query point
            r: Search radius
            
        Returns:
            List of indices of points within radius
        """
        point = np.asarray(point, dtype=np.float64)
        results = []
        
        def _search(node: Optional[KDNode]):
            if node is None:
                return
            
            dist = np.linalg.norm(point - node.point)
            if dist <= r:
                results.append(node.index)
            
            axis = node.axis
            diff = point[axis] - node.point[axis]
            
            if diff <= 0:
                _search(node.left)
                if abs(diff) <= r:
                    _search(node.right)
            else:
                _search(node.right)
                if abs(diff) <= r:
                    _search(node.left)
        
        _search(self.root)
        return results


# ============================================================
# WRAPPER KD-TREE (using scipy)
# ============================================================

class WrapperKDTree:
    """
    Wrapper around scipy's cKDTree with the same interface as SelfKDTree.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Build KD-tree from data points.
        
        Args:
            data: Array of shape (n_points, n_dimensions)
        """
        self.data = np.asarray(data, dtype=np.float64)
        self.n_points, self.n_dims = self.data.shape
        self._tree = cKDTree(self.data)
    
    def query(self, point: np.ndarray, k: int = 1,
              distance_upper_bound: float = float('inf')) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to query point.
        
        Args:
            point: Query point
            k: Number of neighbors to find
            distance_upper_bound: Maximum distance to consider
            
        Returns:
            (distances, indices) arrays of length k
        """
        distances, indices = self._tree.query(
            point, k=k, distance_upper_bound=distance_upper_bound
        )
        return np.atleast_1d(distances), np.atleast_1d(indices)
    
    def query_ball_point(self, point: np.ndarray, r: float) -> List[int]:
        """
        Find all points within radius r of query point.
        """
        return self._tree.query_ball_point(point, r)


# ============================================================
# REPRESENTATIVE TREE (for CURE algorithm)
# ============================================================


class RepresentativeTree:
    """
    A tree structure for storing cluster representative points.
    
    Supports efficient nearest neighbor queries for finding the closest
    cluster to a given query cluster.
    
    For Euclidean distance, uses scipy's cKDTree for efficiency.
    For custom metrics (like Pearson), uses brute-force with optimizations.
    """
    
    def __init__(self, metric: str = "euclidean"):
        """
        Initialize the representative tree.
        
        Args:
            metric: Distance metric - "euclidean" or "pearson"
        """
        self.metric = metric
        self.rep_points: Optional[np.ndarray] = None
        self.rep_to_cluster: Dict[int, int] = {}  # rep_index -> cluster_id
        self.cluster_to_reps: Dict[int, List[int]] = {}  # cluster_id -> [rep_indices]
        self.kdtree: Optional[cKDTree] = None
        
        # For Pearson: precomputed normalized representatives
        self._normalized_reps: Optional[np.ndarray] = None
    
    def build(self, clusters: Dict[int, Any]) -> None:
        """
        Build the tree from active clusters.
        
        Args:
            clusters: Dictionary of cluster_id -> Cluster objects
                     Each cluster must have .reps (list of representative points)
                     and .alive attribute (bool)
        """
        rep_list = []
        self.rep_to_cluster = {}
        self.cluster_to_reps = {}
        
        rep_idx = 0
        for cluster_id, cluster in clusters.items():
            if not cluster.alive:
                continue
            
            self.cluster_to_reps[cluster_id] = []
            for rep in cluster.reps:
                rep_list.append(rep)
                self.rep_to_cluster[rep_idx] = cluster_id
                self.cluster_to_reps[cluster_id].append(rep_idx)
                rep_idx += 1
        
        if not rep_list:
            self.rep_points = None
            self.kdtree = None
            self._normalized_reps = None
            return
        
        self.rep_points = np.array(rep_list)
        
        if self.metric == "euclidean":
            self.kdtree = cKDTree(self.rep_points)
        elif self.metric == "pearson":
            # Precompute normalized representatives for Pearson correlation
            self._precompute_normalized()
    
    def _precompute_normalized(self) -> None:
        """Precompute normalized representatives for Pearson distance."""
        if self.rep_points is None or len(self.rep_points) == 0:
            self._normalized_reps = None
            return
        
        # Center each representative (subtract mean per row)
        centered = self.rep_points - np.mean(self.rep_points, axis=1, keepdims=True)
        # Normalize
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        self._normalized_reps = centered / norms
    
    def find_closest_cluster(
        self, 
        query_cluster: Any, 
        clusters: Dict[int, Any],
        threshold: float = float('inf')
    ) -> Tuple[Optional[int], float]:
        """
        Find the closest cluster to the query cluster using the tree.
        
        This implements Step 15 from the CURE paper:
        closest_cluster(T, x, dist(x, w))
        
        Args:
            query_cluster: The cluster to find neighbors for
            clusters: Dictionary of all clusters
            threshold: Maximum distance to consider (for pruning)
            
        Returns:
            Tuple of (closest_cluster_id, distance) or (None, inf) if not found
        """
        if self.rep_points is None or len(self.rep_points) == 0:
            return None, float('inf')
        
        if self.metric == "euclidean":
            return self._find_closest_euclidean(query_cluster, clusters, threshold)
        else:
            return self._find_closest_pearson(query_cluster, clusters, threshold)
    
    def _find_closest_euclidean(
        self, 
        query_cluster: Any,
        clusters: Dict[int, Any],
        threshold: float
    ) -> Tuple[Optional[int], float]:
        """Find closest cluster using Euclidean distance and KD-tree."""
        min_dist = float('inf')
        closest_cluster_id = None
        
        for query_rep in query_cluster.reps:
            # Query KD-tree for nearest neighbors within threshold
            # Use k=min(10, n) to get multiple candidates
            k = min(10, len(self.rep_points))
            distances, indices = self.kdtree.query(
                query_rep, 
                k=k, 
                distance_upper_bound=threshold
            )
            
            # Handle single result vs multiple
            if np.isscalar(distances):
                distances = [distances]
                indices = [indices]
            
            for dist, idx in zip(distances, indices):
                if dist >= threshold or idx >= len(self.rep_points):
                    continue
                
                neighbor_cluster_id = self.rep_to_cluster.get(idx)
                
                # Skip if same cluster or invalid
                if neighbor_cluster_id is None or neighbor_cluster_id == query_cluster.id:
                    continue
                
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster_id = neighbor_cluster_id
        
        return closest_cluster_id, min_dist
    
    def _find_closest_pearson(
        self, 
        query_cluster: Any,
        clusters: Dict[int, Any],
        threshold: float
    ) -> Tuple[Optional[int], float]:
        """Find closest cluster using Pearson correlation distance."""
        if self._normalized_reps is None:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_cluster_id = None
        
        # Normalize query representatives
        query_reps = np.array(query_cluster.reps)
        query_centered = query_reps - np.mean(query_reps, axis=1, keepdims=True)
        query_norms = np.linalg.norm(query_centered, axis=1, keepdims=True)
        query_norms = np.where(query_norms < 1e-10, 1.0, query_norms)
        query_normalized = query_centered / query_norms
        
        # Compute correlation matrix: (n_query_reps, n_tree_reps)
        corr_matrix = np.dot(query_normalized, self._normalized_reps.T)
        
        # Distance = 1 - correlation
        dist_matrix = 1.0 - corr_matrix
        
        # For each tree representative, find minimum distance from query reps
        for rep_idx in range(len(self.rep_points)):
            cluster_id = self.rep_to_cluster.get(rep_idx)
            
            # Skip if same cluster
            if cluster_id == query_cluster.id:
                continue
            
            # Minimum distance from any query rep to this tree rep
            min_dist_to_rep = np.min(dist_matrix[:, rep_idx])
            
            if min_dist_to_rep < threshold and min_dist_to_rep < min_dist:
                min_dist = min_dist_to_rep
                closest_cluster_id = cluster_id
        
        return closest_cluster_id, min_dist


def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(p - q)


def pearson_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Pearson correlation distance: d = 1 - correlation(p, q)
    
    Returns value in [0, 2] where 0 = identical patterns, 2 = opposite patterns
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
    
    # Pearson correlation = dot product of normalized centered vectors
    correlation = np.dot(p_centered, q_centered) / (p_norm * q_norm)
    
    # Clamp to [-1, 1] for numerical stability
    correlation = np.clip(correlation, -1.0, 1.0)
    
    return 1.0 - correlation


def cluster_distance_euclidean(u: Any, v: Any) -> float:
    """
    Compute distance between two clusters using Euclidean distance.
    
    Distance = minimum Euclidean distance between any pair of representatives.
    """
    min_dist = float('inf')
    for p in u.reps:
        for q in v.reps:
            dist = euclidean_distance(p, q)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def cluster_distance_pearson(u: Any, v: Any) -> float:
    """
    Compute distance between two clusters using Pearson correlation distance.
    
    Distance = minimum Pearson distance between any pair of representatives.
    Uses vectorized computation for efficiency.
    """
    if len(u.reps) == 0 or len(v.reps) == 0:
        return float('inf')
    
    u_reps = np.array(u.reps)
    v_reps = np.array(v.reps)
    
    # Center and normalize
    u_centered = u_reps - np.mean(u_reps, axis=1, keepdims=True)
    v_centered = v_reps - np.mean(v_reps, axis=1, keepdims=True)
    
    u_norms = np.linalg.norm(u_centered, axis=1, keepdims=True)
    v_norms = np.linalg.norm(v_centered, axis=1, keepdims=True)
    
    u_norms = np.where(u_norms < 1e-10, 1.0, u_norms)
    v_norms = np.where(v_norms < 1e-10, 1.0, v_norms)
    
    u_normalized = u_centered / u_norms
    v_normalized = v_centered / v_norms
    
    # Correlation matrix
    corr_matrix = np.dot(u_normalized, v_normalized.T)
    
    # Distance = 1 - max correlation
    return 1.0 - np.max(corr_matrix)
