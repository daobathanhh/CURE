import numpy as np
import heapq

import matplotlib.pyplot as plt
import sys

class Cluster:
    def __init__(self, cluster_id, points, mean, reps=None):
        self.id = cluster_id
        self.points = points  # List of point indices
        self.mean = mean      # numpy array
        self.reps = reps if reps is not None else [] # List of numpy arrays (representatives)
        
        # Precomputed normalized reps for fast Pearson
        self.reps_norm = [] 
        if self.reps:
            self.reps_norm = [self._precompute(r) for r in self.reps]
            
        self.closest = None       # Reference to closest Cluster object
        self.closest_dist = float('inf') # Distance to closest cluster
        self.active = True # Flag to handle lazy deletion in heap/list

    def _precompute(self, vec):
        """Helper to precompute centered and normalized vector for Pearson."""
        v_centered = vec - np.mean(vec)
        norm = np.linalg.norm(v_centered)
        if norm == 0:
            return v_centered # Avoid divide by zero, though mostly handled
        return v_centered / norm

    def add_rep(self, rep):
        self.reps.append(rep)
        self.reps_norm.append(self._precompute(rep))

class CURE:
    def __init__(self, k, c=3, alpha=0.3):
        """
        k: number of clusters
        c: number of representatives
        alpha: shrink factor (0 to 1)
        """
        self.k = k
        self.c = c
        self.alpha = alpha
        self.clusters = {} # id -> Cluster
        self.next_id = 0
        self.heap = [] # Min-heap storing (distance, cluster_id)

    def _pearson_dist(self, u_norm_reps, v_norm_reps):
        """
        Compute CURE distance between two clusters u and v.
        dist(u, v) = 1 - max(pearson(r_u, r_v)) for r_u in u.reps, r_v in v.reps
        
        Args:
           u_norm_reps: List of precomputed (centered/normalized) reps for u
           v_norm_reps: List of precomputed (centered/normalized) reps for v
        """
        max_corr = -1.0
        
        # Optimization: matrix multiplication if possible, but loops are fine for small c
        # Given c is small (e.g., 3-10), double loop is fast enough usually.
        # But we can stack them for numpy speedup if needed.
        
        if not u_norm_reps or not v_norm_reps:
            return float('inf')

        # Vectorized approach
        # U: (c_u, features), V: (c_v, features)
        # Corr = U @ V.T -> (c_u, c_v) matrix
        
        U = np.stack(u_norm_reps)
        V = np.stack(v_norm_reps)
        
        # Dot product of normalized centered vectors is the Pearson correlation
        corrs = np.dot(U, V.T)
        max_corr = np.max(corrs)
            
        return 1.0 - max_corr

    def _dist_point_to_set(self, point_norm, set_norm_reps):
        """
        Distance from a point to a set of points (min distance).
        For CURE merge step representative selection.
        dist(p, Q) = min_{q in Q} dist(p, q)
        """
        if not set_norm_reps:
            return float('inf')
            
        # Point: (d,), Set: (n, d)
        Q = np.stack(set_norm_reps)
        
        # dot product = correlation
        corrs = np.dot(Q, point_norm)
        
        # dist = 1 - corr
        dists = 1.0 - corrs
        return np.min(dists)

    def _merge(self, u, v, data, data_norm):
        """
        Merge clusters u and v into w.
        """
        # 1. Union points
        w_points = u.points + v.points
        
        # 2. Compute mean
        # w.mean <-- (|u|*u.mean + |v|*v.mean) / (|u| + |v|)
        num_u = len(u.points)
        num_v = len(v.points)
        w_mean = (num_u * u.mean + num_v * v.mean) / (num_u + num_v)
        
        w = Cluster(self.next_id, w_points, w_mean)
        self.next_id += 1
        
        # 3. Select representatives
        # tmpSet starts empty
        tmp_reps = [] # Raw representatives
        tmp_reps_norm = [] # Normalized representatives for distance calc
        
        # Pre-filter: We need to iterate over all points in cluster w
        # Optimization: data_norm contains precomputed P for all points
        
        for i in range(self.c):
            max_dist = -1.0
            max_point = None
            max_point_norm = None
            
            # This loop is O(|w|). Can be expensive.
            for p_idx in w_points:
                p_vec = data[p_idx]
                p_norm = data_norm[p_idx]
                
                if i == 0:
                    # dist(p, w.mean)
                    # We need w.mean normalized for Pearson
                    w_mean_norm = w._precompute(w_mean)
                    # dist = 1 - dot
                    min_dist = 1.0 - np.dot(p_norm, w_mean_norm)
                else:
                    # dist = min { dist(p, q) : q in tmpSet }
                    min_dist = self._dist_point_to_set(p_norm, tmp_reps_norm)
                
                if min_dist >= max_dist:
                    max_dist = min_dist
                    max_point = p_vec
                    max_point_norm = p_norm
            
            # If we found a point (should always happen if cluster not empty)
            if max_point is not None:
                tmp_reps.append(max_point)
                tmp_reps_norm.append(max_point_norm)
            else:
                 break # Should not happen unless empty
        
        # 4. Shrink representatives
        # w.rep <-- w.rep U { p + alpha * (w.mean - p) }
        for p in tmp_reps:
            shrink_rep = p + self.alpha * (w_mean - p)
            w.add_rep(shrink_rep)
            
        return w

    def cure(self, data):
        """
        Main clustering procedure.
        data: numpy array (n_samples, n_features)
        """
        data = np.array(data)
        n, d = data.shape
        
        # 0. Precompute normalized data for fast Pearson during merge selection
        data_centered = data - np.mean(data, axis=1, keepdims=True)
        # Avoid zero division
        norms = np.linalg.norm(data_centered, axis=1, keepdims= True)
        norms[norms == 0] = 1.0
        data_norm = data_centered / norms

        # 1. Initialize clusters
        # Each point is a cluster
        self.clusters = {}
        for i in range(n):
            c = Cluster(i, [i], data[i])
            c.add_rep(data[i]) # Initially, the point itself is the rep
            self.clusters[i] = c
        
        self.next_id = n
        
        # 2. Build Heap
        # Naive: Compute all pairwise distances O(n^2)
        # We need efficient lookup.
        
        active_ids = list(self.clusters.keys())
        
        # Calculate initial closest for each cluster
        # This is O(N^2)
        print("Computing initial distance matrix...")
        for i_idx in range(len(active_ids)):
            u_id = active_ids[i_idx]
            u = self.clusters[u_id]
            
            min_dist = float('inf')
            closest_target = None
            
            for j_idx in range(len(active_ids)):
                if i_idx == j_idx:
                    continue
                v_id = active_ids[j_idx]
                v = self.clusters[v_id]
                
                dist = self._pearson_dist(u.reps_norm, v.reps_norm)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_target = v
            
            u.closest = closest_target
            u.closest_dist = min_dist
            
            # Push to heap: (dist, u.id)
            # Use tuple (dist, u.id) to tie-break deterministically
            heapq.heappush(self.heap, (min_dist, u.id))
            
        print("Starting clustering loop...")
        
        # 3. Main Loop
        # WHILE size(Q) > k
        active_count = len(self.clusters)
        
        while active_count > self.k:
            if not self.heap:
                break
                
            # Extract min
            # Lazy deletion: Pop until we find a cluster that is active and distances match
            while True:
                if not self.heap:
                    return # Should not happen
                d_u, u_id = heapq.heappop(self.heap)
                
                if u_id not in self.clusters:
                    continue # Was deleted
                
                u = self.clusters[u_id]
                if not u.active:
                    continue
                    
                # Check if this distance is stale
                # Floating point comparison tolerance maybe needed, but usually exact match if logic is correct
                # Or just check if u.closest_dist roughly equals d_u
                if abs(u.closest_dist - d_u) > 1e-9:
                    # Stale entry, ignore (or re-insert if we were doing generic lazy updates,
                    # but here we only extract the global min)
                    # Actually, if u's closest changed, we updated u.closest_dist.
                    # So if d_u != u.closest_dist, this heap entry is old.
                    continue
                
                break
            
            # u is the cluster to merge
            v = u.closest
            
            # Check if v is still valid
            if v is None or v.id not in self.clusters or not v.active:
                # v was removed essentially.
                # We need to recompute u's closest and push back.
                # This can happen if v was merged into something else previously?
                # Actually, if v was merged, we should have updated u.closest then.
                # But let's be safe.
                self._update_closest(u)
                heapq.heappush(self.heap, (u.closest_dist, u.id))
                continue

            # Check if v is also "ready" to be merged with u? 
            # The algorithm says: u = extract_min, v = u.closest. Simply merge.
            
            # Merge u and v
            w = self._merge(u, v, data, data_norm)
            
            # Delete u and v
            u.active = False
            v.active = False
            del self.clusters[u.id]
            del self.clusters[v.id]
            active_count -= 2
            
            # Insert w
            self.clusters[w.id] = w
            active_count += 1
            
            # w.closest <-- x (arbitrary)
            # We will compute w.closest correctly in the loop below or specifically
            # But the pseudocode loop essentially does two things:
            # 1. Update w's closest
            # 2. Update others' closest if it needs to be w, or if it WAS u/v
            
            # Optimization: We just iterate all active clusters once.
            
            w_min_dist = float('inf')
            w_closest = None
            
            # We iterate over a snapshot of keys because we might modify heap? No, heap is separate.
            # But specific clusters dict is modified.
            current_cluster_ids = list(self.clusters.keys())
            
            for x_id in current_cluster_ids:
                if x_id == w.id:
                    continue
                    
                x = self.clusters[x_id]
                
                # dist(w, x)
                dist_wx = self._pearson_dist(w.reps_norm, x.reps_norm)
                
                # Update w's closest
                if dist_wx < w_min_dist:
                    w_min_dist = dist_wx
                    w_closest = x
                    
                # Update x's closest
                
                # Case 1: x's closest was u or v
                # We forced to recheck scans
                # BUT, we can simplify logic: just compare dist_wx with current x.closest
                # IF x.closest was u or v, we MUST re-scan everything for x later?
                # The pseudocode says:
                # IF x.closest is u or v THEN
                #    IF dist(x, x.closest) < dist(x, w) ... wait, x.closest is u/v, so dist(x, u) or dist(x, v)
                #    This part of pseudocode is tricky. "x.closest" is the POINTER. 
                #    If it points to deleted object, valid distance is invalid.
                #    So if x.closest was u or v, we essentially lost our knowledge of x's nearest neighbor.
                #    We have to find x's new nearest neighbor. It *might* be w, or might be some other z.
                
                needs_update = False
                
                if x.closest is u or x.closest is v:
                    # We need to find new closest for x from scratch (among all active + w)
                    # We can't assume w is the closest.
                    # Pseudocode optimization:
                    # x.closest <- closest_cluster(T, x, dist(x, w)) <-- KDTree optimization
                    # Since we don't have KDTree, we scan all.
                    self._update_closest(x, candidate_w=w, candidate_w_dist=dist_wx)
                    needs_update = True
                else:
                     # x.closest is some valid cluster z != u, v
                     # Check if w is closer than z
                     if dist_wx < x.closest_dist:
                         x.closest = w
                         x.closest_dist = dist_wx
                         needs_update = True
                         
                if needs_update:
                    heapq.heappush(self.heap, (x.closest_dist, x.id))
            
            # Finally set w properties
            w.closest = w_closest
            w.closest_dist = w_min_dist
            heapq.heappush(self.heap, (w.closest_dist, w.id))
            
            # print(f"Clusters: {active_count}, Merged {u.id} & {v.id} -> {w.id} (Dist: {u.closest_dist:.4f})")

    def _update_closest(self, x, candidate_w=None, candidate_w_dist=None):
        """
        Full scan to find closest to x.
        If candidate_w is provided, we can consider it as a starter.
        """
        min_dist = float('inf')
        best_target = None
        
        if candidate_w:
            min_dist = candidate_w_dist
            best_target = candidate_w
        
        for y_id, y in self.clusters.items():
            if y_id == x.id:
                continue
            
            # If we already have a candidate W, and we are iterating, we check if Y is closer.
            if best_target and y_id == best_target.id:
                continue
                
            d = self._pearson_dist(x.reps_norm, y.reps_norm)
            if d < min_dist:
                min_dist = d
                best_target = y
        
        x.closest = best_target
        x.closest_dist = min_dist

    def get_labels(self):
        # Map original indices to cluster labels
        labels = {}
        for c_id, c in self.clusters.items():
            for p_idx in c.points:
                labels[p_idx] = c_id
        
        # Return list ordered by index
        sorted_indices = sorted(labels.keys())
        return [labels[i] for i in sorted_indices]

    def get_colors(self, num_clusters):
        cmap = plt.colormaps.get_cmap('viridis')
        return [cmap(i) for i in np.linspace(0, 1, num_clusters)]   

    def plot_2d_clusters(self, labels, title="2D Scatter Plot of Clusters"):
        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(unique_labels):
            cluster_points = self.S[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[i], label=f'Cluster {label}', 
                        s=50, alpha=0.8)
        plt.title(title)
        plt.xlabel(f'Feature 1 (Axis 0)')
        plt.ylabel(f'Feature 2 (Axis 1)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_3d_clusters(self, labels, title="3D Scatter Plot of Clusters"):
        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(unique_labels):
            cluster_points = self.S[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                    color=colors[i], label=f'Cluster {label}', 
                    s=60, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()
        plt.show() 

    def plot_high_dim_pca(self, labels, title="PCA Projection (D > 3)"):
        """
        Reduces high-dimensional data (D > 3) to 2 principal components 
        and plots the result.
        """
        pca = PCA(n_components=2)
        S_2d = pca.fit_transform(self.S)

        unique_labels = np.unique(labels)
        colors = self.get_colors(len(unique_labels))

        plt.figure(figsize=(8, 6))
        
        for i, label in enumerate(unique_labels):
            cluster_points = S_2d[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[i], label=f'Cluster {label}', 
                        s=50, alpha=0.8)

        explained_variance = pca.explained_variance_ratio_.sum()
        
        plt.title(f"{title}\nExplained Variance: {explained_variance:.2f}")
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_labels_from_clusters(self, final_clusters):
        N = len(self.S)
        labels = np.zeros(N, dtype=int)
        for cluster_id, cluster in enumerate(final_clusters):
            for point_index in cluster.points:
                labels[point_index] = cluster_id
        return labels

    def visualize(self, final_clusters, algorithm_name="CURE"):
        """
        Determines the dimensionality of the data and calls the appropriate 
        visualization function.
        """
        labels = self.get_labels_from_clusters(final_clusters)
        D = self.S.shape[1]
        
        if D == 2:
            self.plot_2d_clusters(labels, title="2D Scatter Plot of Clusters")
        elif D == 3:
            self.plot_3d_clusters(labels, title="3D Scatter Plot of Clusters")
            self.plot_high_dim_pca(labels, f"{algorithm_name} PCA Projection (D={D})")
        elif D > 3:
            self.plot_high_dim_pca(labels, f"{algorithm_name} PCA Projection (D={D})")
        else:
            print("Data dimension is 1. Visualization not implemented.")



import numpy as np
S = np.array([[82.68456706,  4.45922012],
       [15.86935568, 40.86335396],
       [61.91947832, 83.62464105],
       [78.5321037 , 55.72687319],
       [35.22380287, 69.52947118],
       [14.31433367, 84.51472094],
       [45.56225679, 86.70398922],
       [38.17813064, 36.24561558],
       [25.12518629, 48.06771104],
       [38.79577979, 69.8433773 ],
       [53.89180384,  8.4574877 ],
       [57.96463358,  2.51607491],
       [19.87958796, 25.00048336],
       [29.60111534, 63.85470859],
       [96.26315396, 75.7944596 ],
       [45.6781928 , 48.01700787],
       [47.36043981, 65.33336647],
       [14.79387894, 53.93647558],
       [ 0.5349268 , 73.8408741 ],
       [ 1.9565076 , 38.83379104],
       [ 7.22259619,  7.46329912],
       [83.70908182, 87.19282854],
       [99.46010373, 88.06434118],
       [10.58629585, 58.17182678],
       [91.16103317, 81.97991079],
       [52.33079582, 36.95667037],
       [16.39406002, 59.02514162],
       [91.25682157, 47.886654  ],
       [59.75783353, 99.76065577],
       [30.12172653, 51.64006229],
       [87.04887465, 19.54581896],
       [35.4454102 , 46.78774647],
       [32.11390573, 81.20586971],
       [ 4.71182292, 36.74864897],
       [ 3.56895639, 52.79824126],
       [20.91179372, 28.06152283],
       [26.71986957, 88.80691215],
       [ 9.83678778, 67.92188701],
       [54.93751982, 43.50173492],
       [89.75902282, 61.76181973],
       [68.22378666, 60.52227261],
       [82.83370095, 52.96478617],
       [81.74331837, 93.72689515],
       [16.90881705, 52.60839024],
       [73.53944709, 69.68580462],
       [69.43734034, 21.50686882],
       [15.41377231, 52.82975414],
       [46.35854535, 53.46012437],
       [15.79851879, 51.07157605],
       [16.33483227, 15.27623169],
       [95.62494752, 65.59904866],
       [60.98646791, 12.5120598 ],
       [97.87091774, 60.25955038],
       [23.25268963, 47.88283382],
       [66.98653963, 45.82274375],
       [24.50001518, 12.27595334],
       [90.62482297, 26.10163377],
       [26.70519833, 29.97363842],
       [17.50240288, 16.19866068],
       [14.62325911, 55.85259812],
       [53.93148459, 99.97245606],
       [64.37868097, 93.79043934],
       [35.21903809, 32.54456791],
       [56.98641416, 14.63091507],
       [47.58180387, 82.16518798],
       [56.53993328, 21.09958963],
       [89.26654502, 44.93145293],
       [46.45184173, 37.40672241],
       [66.12699612, 25.57541456],
       [29.00718138,  1.33773216],
       [87.0515256 , 93.26402434],
       [32.15622127, 78.43082708],
       [94.28719189,  7.52011228],
       [14.20634369,  6.98549156],
       [53.46566368, 75.99886882],
       [ 9.95331317,  2.71253542],
       [58.7819291 , 29.15052768],
       [79.06911821, 15.86612444],
       [43.31852003, 50.16360299],
       [58.52516723, 73.98061493],
       [84.60479974, 47.95630807],
       [ 4.40048228, 44.45135353],
       [73.48815099, 21.85374176],
       [21.2640658 , 15.84101504],
       [55.26823867, 82.68339427],
       [86.39822125, 64.03112123],
       [14.0346749 , 72.51956147],
       [69.67670801, 17.69164319],
       [99.25664713, 70.12692082],
       [35.05569725, 69.25629184],
       [ 7.65597509, 65.68552145],
       [51.19585098, 95.57447799],
       [47.75689988, 19.49903901],
       [68.53747516,  9.1824247 ],
       [18.96192323, 18.01509825],
       [63.17996916, 10.61488992],
       [68.00907032, 62.11248417],
       [36.62704183, 72.57986598],
       [88.28433514, 30.50844673],
       [84.55027487, 61.67768895],
       [59.94312233, 12.63362876],
       [31.69575909, 90.27180413],
       [65.23166435,  6.20209849],
       [18.04401365, 67.39479239],
       [ 6.95013218, 11.01505727],
       [69.2557347 , 22.88141242],
       [29.04451722,  8.95596388],
       [37.3877622 , 37.53720104],
       [75.32597049, 93.23091484],
       [28.76642235, 87.88358978],
       [82.06950295, 98.32890762],
       [46.51349181, 90.51103712],
       [61.20806614, 80.32981976],
       [68.73248806,  2.26841179],
       [36.30248833, 52.77997883],
       [32.80198452, 89.83846476],
       [81.1819152 , 63.65975624],
       [83.95030926, 12.06409136],
       [91.51060964, 86.11196053],
       [16.98574816, 42.20737922],
       [18.45941299, 91.4280678 ],
       [ 7.99552013, 20.51448468],
       [68.05039452, 92.56722853],
       [26.06795327, 81.69248808],
       [92.38489181, 21.74713364],
       [69.94215954, 14.56188632],
       [19.69490657, 17.37566407],
       [89.10269625, 39.23804332],
       [70.92497779, 28.64422886],
       [91.43405098,  2.5601724 ],
       [39.32875024, 13.52380743],
       [88.57709428, 36.94695247],
       [20.23036818, 98.86408233],
       [43.84770104, 21.74923664],
       [46.96982643, 37.32634408],
       [14.08860497, 36.36306382],
       [49.39713726, 13.88859656],
       [ 9.72428221, 41.82910643],
       [33.55110921, 74.07535433],
       [19.21161401, 84.95056794],
       [ 2.05857404, 21.26401099],
       [65.56571167, 37.06532122],
       [44.74177903, 72.74861354],
       [ 5.85473031,  1.0007354 ],
       [66.62568178, 75.35017042],
       [46.97446318, 33.20796734],
       [88.25275129, 47.57305316],
       [37.16209226, 99.4801409 ],
       [96.48200633, 99.33198113],
       [16.24682867, 81.87817515],
       [60.93794619, 67.17168511],
       [33.34482629, 99.24595876],
       [75.50400476,  7.44998757],
       [41.36792183,  9.77071218],
       [45.18217333, 43.04981032],
       [19.9093177 , 79.63131394],
       [14.14785596, 43.49740298],
       [30.87305779, 92.15495235],
       [35.38365041, 58.00158456],
       [91.65797048, 24.12955327],
       [62.92176766, 54.11590603],
       [81.17907883, 81.70157175],
       [30.16774745, 99.80564807],
       [95.65703792, 75.65790447],
       [ 5.15771472, 74.01281953],
       [21.65343839, 97.08906101],
       [52.49049315, 86.72663471],
       [ 6.82667805,  3.75871534],
       [11.95860806, 86.02196542],
       [ 4.89411092, 43.47584336],
       [67.62348882, 11.7502263 ],
       [46.39875405, 68.02804694],
       [53.20598591, 34.0089533 ],
       [95.52395479, 57.24704101],
       [55.4862418 , 10.95895732],
       [81.15931526, 46.88664817],
       [56.69246413, 74.17423445],
       [98.54556888, 31.02732811],
       [20.89407524, 22.02683976],
       [58.11174018, 65.28731387],
       [31.83236563, 49.23416373],
       [74.77591598, 47.17096975],
       [12.02816917, 52.77834219],
       [ 6.19772266, 29.79945563],
       [ 2.61194772, 47.8387975 ],
       [30.94038661, 31.94677585],
       [76.18204171, 79.27624062],
       [53.33680115, 74.06711373],
       [89.87049934, 58.61213401],
       [87.58609509, 94.56424317],
       [24.12137522, 37.42993941],
       [75.24281553,  5.22281284],
       [15.34360843, 71.91140591],
       [54.05147419, 92.24689086],
       [78.74153397, 92.90997195],
       [63.00926935, 95.19819928],
       [70.00561018, 78.93404605],
       [48.7853465 , 58.83323314],
       [ 4.09693836, 62.35609454],
       [66.50375609, 40.8561185 ]])

if __name__ == "__main__":
    # Example usage / Demo
    print("Running CURE demo inside new_cure.py...")
    
    # Generate simple synthetic data
    # np.random.seed(42)
    # # 3 clusters in 2D
    # c1 = np.random.randn(50, 2) + [5, 5]
    # c2 = np.random.randn(50, 2) + [5, -5]
    # c3 = np.random.randn(50, 2) + [-5, 0]
    # data = np.vstack([c1, c2, c3])
    
    # Run CURE
    cure = CURE(k=3, c=5, alpha=0.3)
    cure.cure(S)

    final_clusters = list(cure.clusters.values())
    labels = cure.get_labels()

    for cl in final_clusters:
        pts = S[cl.points]
        print(cl.id, pts.shape)

    cure.S = S
    cure.visualize(final_clusters)
    
    print(f"Done. Found {len(np.unique(labels))} clusters.")
