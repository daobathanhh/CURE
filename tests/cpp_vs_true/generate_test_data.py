#!/usr/bin/env python3
"""
Generate test data for C++ vs true-labels-only testing.

Same test cases as tests/comparison/generate_all_variants.py (test1_small_2d
through test7_pattern_15d) so results and plots are comparable. Writes
tests/cpp_vs_true/test_data/<name>/ with data.csv, true_labels.csv, params.json.
No Python CURE is run. After running ./build/tests/test_cpp_vs_true, use
plot_cpp_vs_true.py to plot True | C++ Euclidean | C++ Pearson.
"""

import numpy as np
import json
import os


def generate_clustered_2d(n_per_cluster, n_clusters, cluster_std=1.0, seed=42):
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        cx, cy = (c % 3) * 10.0, (c // 3) * 10.0
        for _ in range(n_per_cluster):
            data.append([cx + np.random.randn() * cluster_std, cy + np.random.randn() * cluster_std])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_clustered_3d(n_per_cluster, n_clusters, cluster_std=1.0, seed=42):
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        cx = (c % 2) * 8.0
        cy = ((c // 2) % 2) * 8.0
        cz = (c // 4) * 8.0
        for _ in range(n_per_cluster):
            data.append([
                cx + np.random.randn() * cluster_std,
                cy + np.random.randn() * cluster_std,
                cz + np.random.randn() * cluster_std,
            ])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_circle_2d(n_per_cluster, n_clusters, radius=5.0, noise=0.3, seed=42):
    """2D: each cluster is a ring/circle (different radius or angle band)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        r = radius * (0.8 + 0.4 * c)
        for _ in range(n_per_cluster):
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta) + np.random.randn() * noise
            y = r * np.sin(theta) + np.random.randn() * noise
            data.append([x, y])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_moons_2d(n_per_cluster, seed=42):
    """2D: two crescent moons (2 clusters)."""
    np.random.seed(seed)
    n = n_per_cluster * 2
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.cos(t) + 0.25 * np.random.randn(n // 2)
    y1 = np.sin(t) + 0.25 * np.random.randn(n // 2)
    x2 = 1 - np.cos(t) + 0.25 * np.random.randn(n // 2)
    y2 = 0.5 - np.sin(t) + 0.25 * np.random.randn(n // 2)
    data = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    true_labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return data, true_labels


def generate_spiral_2d(n_per_cluster, n_clusters=3, noise=0.4, seed=42):
    """2D: spiral arms (each arm = one cluster)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        t = np.linspace(0, 3 * np.pi, n_per_cluster) + c * 2 * np.pi / n_clusters
        r = 0.5 + 0.3 * t
        x = r * np.cos(t) + np.random.randn(n_per_cluster) * noise
        y = r * np.sin(t) + np.random.randn(n_per_cluster) * noise
        for i in range(n_per_cluster):
            data.append([x[i], y[i]])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_sphere_shell_3d(n_per_cluster, n_clusters, radius=4.0, thickness=0.6, seed=42):
    """3D: clusters on different spherical shells (same center)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        r_base = radius * (0.6 + 0.35 * c)
        for _ in range(n_per_cluster):
            phi = np.random.uniform(0, np.pi)
            theta = np.random.uniform(0, 2 * np.pi)
            r = r_base + np.random.randn() * thickness
            r = max(0.5, r)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            data.append([x, y, z])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_spiral_3d(n_per_cluster, n_clusters=3, radius=1.0, height=2.0, noise=0.15, seed=42):
    """3D: helical spirals (each helix = one cluster)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        t = np.linspace(0, 4 * np.pi, n_per_cluster) + c * 2 * np.pi / n_clusters
        x = radius * np.cos(t) + np.random.randn(n_per_cluster) * noise
        y = radius * np.sin(t) + np.random.randn(n_per_cluster) * noise
        z = height * t / (4 * np.pi) + np.random.randn(n_per_cluster) * noise
        for i in range(n_per_cluster):
            data.append([x[i], y[i], z[i]])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_torus_3d(n_per_cluster, n_clusters=2, R=3.0, r=1.0, noise=0.2, seed=42):
    """3D: points on torus (two clusters = inner/outer or two segments)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, 2 * np.pi) + c * np.pi  # shift for second cluster
            x = (R + r * np.cos(v)) * np.cos(u) + np.random.randn() * noise
            y = (R + r * np.cos(v)) * np.sin(u) + np.random.randn() * noise
            z = r * np.sin(v) + np.random.randn() * noise
            data.append([x, y, z])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_ellipses_2d(n_per_cluster, n_clusters=3, a=4.0, b=1.5, noise=0.25, seed=42):
    """2D: each cluster is an ellipse with different orientation."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        angle = c * 2 * np.pi / n_clusters
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for _ in range(n_per_cluster):
            theta = np.random.uniform(0, 2 * np.pi)
            x0 = a * np.cos(theta) + np.random.randn() * noise
            y0 = b * np.sin(theta) + np.random.randn() * noise
            x = cos_a * x0 - sin_a * y0 + 3 * (c % 2) - 1.5
            y = sin_a * x0 + cos_a * y0 + 3 * (c // 2) - 1.5
            data.append([x, y])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_annulus_2d(n_inner=50, n_outer=80, r_inner=2.0, r_outer=5.0, noise=0.3, seed=42):
    """2D: two concentric rings (annulus): inner ring + outer ring."""
    np.random.seed(seed)
    data, true_labels = [], []
    for _ in range(n_inner):
        theta = np.random.uniform(0, 2 * np.pi)
        r = r_inner + np.random.randn() * noise
        r = max(0.5, r)
        data.append([r * np.cos(theta), r * np.sin(theta)])
        true_labels.append(0)
    for _ in range(n_outer):
        theta = np.random.uniform(0, 2 * np.pi)
        r = r_outer + np.random.randn() * noise
        r = max(r_inner + 0.5, r)
        data.append([r * np.cos(theta), r * np.sin(theta)])
        true_labels.append(1)
    return np.array(data), np.array(true_labels)


def generate_scurve_2d(n_per_cluster, n_clusters=3, noise=0.2, seed=42):
    """2D: S-curves (each cluster = one branch of stretched S)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        t = np.linspace(0, np.pi, n_per_cluster) + c * np.pi / 2
        x = np.sin(t) + np.random.randn(n_per_cluster) * noise
        y = 2 * np.cos(t) + c * 2.5 + np.random.randn(n_per_cluster) * noise
        for i in range(n_per_cluster):
            data.append([x[i], y[i]])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_cross_2d(n_per_arm=35, arm_length=4.0, noise=0.25, seed=42):
    """2D: cross/plus shape (4 arms from center = 4 clusters)."""
    np.random.seed(seed)
    data, true_labels = [], []
    arms = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for c, (dx, dy) in enumerate(arms):
        for _ in range(n_per_arm):
            t = np.random.uniform(0.3, arm_length)
            x = dx * t + np.random.randn() * noise
            y = dy * t + np.random.randn() * noise
            data.append([x, y])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_cylinder_3d(n_per_cluster, n_clusters=3, radius=2.0, height=4.0, noise=0.2, seed=42):
    """3D: vertical cylinder; clusters at different height bands."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        z_base = height * (c / n_clusters) + height / (2 * n_clusters)
        for _ in range(n_per_cluster):
            theta = np.random.uniform(0, 2 * np.pi)
            r = radius + np.random.randn() * noise
            r = max(0.3, r)
            z = z_base + np.random.randn() * (height / n_clusters * 0.4)
            data.append([r * np.cos(theta), r * np.sin(theta), z])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_cone_3d(n_per_cluster, n_clusters=2, radius=3.0, height=4.0, noise=0.2, seed=42):
    """3D: cone; two clusters = bottom ring vs upper cone surface."""
    np.random.seed(seed)
    data, true_labels = [], []
    # Bottom disc
    for _ in range(n_per_cluster):
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1)) + np.random.randn() * noise
        r = max(0, r)
        data.append([r * np.cos(theta), r * np.sin(theta), 0])
        true_labels.append(0)
    # Upper cone surface
    for _ in range(n_per_cluster):
        h = np.random.uniform(0.2, 1.0) * height
        r = radius * (1 - h / height) + np.random.randn() * noise
        r = max(0.2, r)
        theta = np.random.uniform(0, 2 * np.pi)
        data.append([r * np.cos(theta), r * np.sin(theta), h])
        true_labels.append(1)
    return np.array(data), np.array(true_labels)


def generate_two_rings_3d(n_per_cluster=60, d=2.5, noise=0.2, seed=42):
    """3D: two rings in perpendicular planes (interlocking)."""
    np.random.seed(seed)
    data, true_labels = [], []
    # Ring in xy-plane
    for _ in range(n_per_cluster):
        theta = np.random.uniform(0, 2 * np.pi)
        x = d * np.cos(theta) + np.random.randn() * noise
        y = d * np.sin(theta) + np.random.randn() * noise
        z = np.random.randn() * noise
        data.append([x, y, z])
        true_labels.append(0)
    # Ring in xz-plane
    for _ in range(n_per_cluster):
        theta = np.random.uniform(0, 2 * np.pi)
        x = d * np.cos(theta) + np.random.randn() * noise
        z = d * np.sin(theta) + np.random.randn() * noise
        y = np.random.randn() * noise
        data.append([x, y, z])
        true_labels.append(1)
    return np.array(data), np.array(true_labels)


def generate_cube_corners_3d(n_per_corner=40, half=2.0, spread=0.6, seed=42):
    """3D: 4 clusters at 4 corners of a cube (opposite corners = same cluster for 2 clusters)."""
    np.random.seed(seed)
    corners = [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1)]
    data, true_labels = [], []
    for c, (sx, sy, sz) in enumerate(corners):
        for _ in range(n_per_corner):
            x = half * sx + np.random.randn() * spread
            y = half * sy + np.random.randn() * spread
            z = half * sz + np.random.randn() * spread
            data.append([x, y, z])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_4d_hypersphere(n_per_cluster, n_clusters=3, radius=2.0, noise=0.2, seed=42):
    """4D: points on hypersphere surface; clusters = different 'caps' (first coord sign + sector)."""
    np.random.seed(seed)
    dim = 4
    data, true_labels = [], []
    np.random.seed(seed)
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            u = np.random.randn(dim)
            u = u / (np.linalg.norm(u) + 1e-10)
            # Rotate cluster to different sector
            angle = c * 2 * np.pi / n_clusters
            u0 = u[0] * np.cos(angle) - u[1] * np.sin(angle)
            u1 = u[0] * np.sin(angle) + u[1] * np.cos(angle)
            u = np.array([u0, u1, u[2], u[3]])
            r = radius + np.random.randn() * noise
            r = max(0.5, r)
            data.append((r * u).tolist())
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_4d_spiral(n_per_cluster=50, n_clusters=3, radius=1.5, seed=42):
    """4D: spiral in 4D (two main dims spiral, other two follow)."""
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        t = np.linspace(0, 3 * np.pi, n_per_cluster) + c * 2 * np.pi / n_clusters
        r = radius * (0.5 + 0.5 * t / (3 * np.pi))
        x1 = r * np.cos(t)
        x2 = r * np.sin(t)
        x3 = 0.5 * t
        x4 = 0.3 * np.sin(2 * t)
        for i in range(n_per_cluster):
            data.append([x1[i], x2[i], x3[i], x4[i]])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_4d_special(n_per_cluster, n_clusters=3, radius=3.0, spread=1.2, seed=42):
    """4D: clusters at different 4D 'corners' with spherical blobs (special structure)."""
    np.random.seed(seed)
    data, true_labels = [], []
    dim = 4
    # Place cluster centers along diagonal and anti-diagonal
    centers = [
        np.array([radius, radius, radius, radius]),
        np.array([-radius, -radius, radius, radius]),
        np.array([radius, -radius, -radius, radius]),
    ]
    for c in range(n_clusters):
        center = centers[c % len(centers)]
        for _ in range(n_per_cluster):
            u = np.random.randn(dim)
            u = u / (np.linalg.norm(u) + 1e-10)
            r = spread * np.abs(np.random.randn())
            pt = center + r * u
            data.append(pt.tolist())
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_pattern_data(n_per_cluster, n_clusters, n_features=10, noise=0.2, seed=42):
    np.random.seed(seed)
    patterns = []
    for c in range(n_clusters):
        p = np.zeros(n_features)
        for i in range(n_features):
            if c == 0:
                p[i] = np.sin(2.0 * np.pi * i / n_features)
            elif c == 1:
                p[i] = np.cos(2.0 * np.pi * i / n_features)
            else:
                p[i] = float(i) / n_features
        patterns.append(p)
    data, true_labels = [], []
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            data.append((patterns[c] + np.random.randn(n_features) * noise).tolist())
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(base_dir, exist_ok=True)

    # Same cases as tests/comparison/generate_all_variants.py
    cases = []

    # 1. Small 2D
    data, true_labels = generate_clustered_2d(30, 3, cluster_std=0.5, seed=42)
    cases.append(('test1_small_2d', data, true_labels, {'k': 3, 'c': 3, 'alpha': 0.3}))

    # 2. Medium 2D
    data, true_labels = generate_clustered_2d(50, 4, cluster_std=1.0, seed=123)
    cases.append(('test2_medium_2d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    # 3. Alpha 0.5
    data, true_labels = generate_clustered_2d(40, 3, cluster_std=0.8, seed=456)
    cases.append(('test3_alpha05', data, true_labels, {'k': 3, 'c': 4, 'alpha': 0.5}))

    # 4. Pearson-style pattern 10D
    data, true_labels = generate_pattern_data(50, 3, n_features=10, noise=0.2, seed=789)
    cases.append(('test4_pattern_10d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # 5. Larger 2D
    data, true_labels = generate_clustered_2d(100, 5, cluster_std=1.5, seed=999)
    cases.append(('test5_large_2d', data, true_labels, {'k': 5, 'c': 5, 'alpha': 0.3}))

    # 6. Overlapping
    np.random.seed(1234)
    data, true_labels = [], []
    for c, (cx, cy) in enumerate([(0, 0), (6, 0), (3, 5)]):
        for _ in range(50):
            data.append([cx + np.random.randn() * 1.3, cy + np.random.randn() * 1.3])
            true_labels.append(c)
    data, true_labels = np.array(data), np.array(true_labels)
    cases.append(('test6_overlapping', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # 7. Pattern 15D, 4 clusters
    data, true_labels = generate_pattern_data(40, 4, n_features=15, noise=0.15, seed=888)
    cases.append(('test7_pattern_15d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    # Special shapes 2D
    data, true_labels = generate_circle_2d(40, 3, radius=5.0, noise=0.3, seed=201)
    cases.append(('shape_circle_2d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_moons_2d(50, seed=202)
    cases.append(('shape_moons_2d', data, true_labels, {'k': 2, 'c': 4, 'alpha': 0.3}))
    data, true_labels = generate_spiral_2d(45, n_clusters=3, noise=0.35, seed=203)
    cases.append(('shape_spiral_2d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # Special shapes 3D
    data, true_labels = generate_sphere_shell_3d(50, 3, radius=4.0, thickness=0.5, seed=301)
    cases.append(('shape_sphere_3d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_spiral_3d(50, n_clusters=3, radius=1.5, height=3.0, noise=0.12, seed=302)
    cases.append(('shape_spiral_3d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_torus_3d(60, n_clusters=2, R=3.0, r=1.0, noise=0.2, seed=303)
    cases.append(('shape_torus_3d', data, true_labels, {'k': 2, 'c': 4, 'alpha': 0.3}))

    # Special shape 4D (plotted via PCA)
    data, true_labels = generate_4d_special(45, n_clusters=3, seed=401)
    cases.append(('shape_4d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # More special shapes 2D
    data, true_labels = generate_ellipses_2d(45, n_clusters=3, a=3.5, b=1.2, noise=0.25, seed=204)
    cases.append(('shape_ellipses_2d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_annulus_2d(50, 70, r_inner=2.0, r_outer=5.0, noise=0.3, seed=205)
    cases.append(('shape_annulus_2d', data, true_labels, {'k': 2, 'c': 4, 'alpha': 0.3}))
    data, true_labels = generate_scurve_2d(45, n_clusters=3, noise=0.2, seed=206)
    cases.append(('shape_scurve_2d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_cross_2d(n_per_arm=40, arm_length=4.0, noise=0.25, seed=207)
    cases.append(('shape_cross_2d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    # More special shapes 3D
    data, true_labels = generate_cylinder_3d(50, n_clusters=3, radius=2.0, height=4.0, noise=0.2, seed=304)
    cases.append(('shape_cylinder_3d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_cone_3d(55, radius=2.5, height=3.5, noise=0.2, seed=305)
    cases.append(('shape_cone_3d', data, true_labels, {'k': 2, 'c': 4, 'alpha': 0.3}))
    data, true_labels = generate_two_rings_3d(60, d=2.5, noise=0.2, seed=306)
    cases.append(('shape_two_rings_3d', data, true_labels, {'k': 2, 'c': 4, 'alpha': 0.3}))
    data, true_labels = generate_cube_corners_3d(40, half=2.0, spread=0.5, seed=307)
    cases.append(('shape_cube_corners_3d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    # More special shapes 4D
    data, true_labels = generate_4d_hypersphere(45, n_clusters=3, radius=2.0, noise=0.2, seed=402)
    cases.append(('shape_4d_hypersphere', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))
    data, true_labels = generate_4d_spiral(50, n_clusters=3, radius=1.5, seed=403)
    cases.append(('shape_4d_spiral', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # 3D–20D: spatial 3D + pattern 4D–20D
    data, true_labels = generate_clustered_3d(40, 4, cluster_std=0.8, seed=100)
    cases.append(('test_3d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    for d in range(4, 21):
        n_clusters = 3 if d <= 8 else 4
        n_pts = 45 if d <= 10 else 40
        noise = 0.2 if d <= 10 else 0.15
        data, true_labels = generate_pattern_data(
            n_pts, n_clusters, n_features=d, noise=noise, seed=800 + d
        )
        cases.append((f'test_{d}d', data, true_labels, {
            'k': n_clusters, 'c': 5, 'alpha': 0.3
        }))

    print('Generating cpp_vs_true test data (same cases as comparison + 3D–20D)...')
    print('=' * 60)
    for name, data, true_labels, params in cases:
        out_dir = os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, 'data.csv'), data, delimiter=',', fmt='%.10f')
        np.savetxt(os.path.join(out_dir, 'true_labels.csv'), true_labels, delimiter=',', fmt='%d')
        with open(os.path.join(out_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  {name}: {len(data)} pts, {data.shape[1]}D, k={params['k']}")
    print('=' * 60)
    print(f'Saved to {base_dir}/')
    print('Run C++: ./build/tests/test_cpp_vs_true')
    print('Plot:    python3 tests/cpp_vs_true/plot_cpp_vs_true.py [--out tests/cpp_vs_true/plots] [--show]')


if __name__ == '__main__':
    main()
