# Scale-Invariant CURE Clustering via Pearson Distance and Multi-Level Outlier-Resistant Sampling for Large-Scale Customer Segmentation

**Authors:** [To be added]  
**Keywords:** hierarchical clustering, CURE, Pearson correlation distance, feature dominance, outlier-resistant sampling, MapReduce, RFM segmentation

---

## Abstract

Distance-based clustering algorithms applied to unstandardized multi-scale data systematically produce degenerate partitions in which one or two clusters absorb the overwhelming majority of points, a phenomenon we term *monopolized clustering*. This failure mode arises from feature dominance: when attributes span vastly different numerical ranges, a single high-magnitude feature controls all pairwise distances, rendering the remaining features irrelevant. Although feature standardization is the conventional remedy, it transforms cluster descriptions into z-score space, destroying the business interpretability of original-unit segment profiles. We investigate this problem empirically within the CURE hierarchical clustering framework applied to RFM (Recency, Frequency, Monetary) customer segmentation, where raw monetary values dominate Euclidean distances by a factor of up to 3,524x over recency features. Our investigation centers on three coordinated adaptations: substituting Pearson correlation distance for Euclidean in CURE, which eliminates feature dominance while preserving feature units; replacing uniform random sampling with centroid-proximity sampling in CURE's large-scale variant, adapting the trimmed-core sampling principle from BIRCH to an agglomerative context; and introducing a tunable *outlier fraction* parameter for distributed MapReduce CURE, which provides per-mapper trimming analogous to trimmed k-means in a distributed setting. We develop clustering evaluation metrics adapted for Pearson distance space, discuss their conceptual limitations, and validate their correlation with clustering balance. Experiments on four real-world RFM datasets (2,500 to 1,283,707 customers) demonstrate that Pearson distance eliminates monopolized clustering in 55-82% of configurations relative to Euclidean, that distributed trimming with outlier fraction f = 0.7-0.8 substantially reduces Top-2 concentration by 14-44 percentage points (with near-uniform balance on datasets with well-separated cluster structure, and significant but incomplete balance improvement on the largest dataset), and that the resulting clusters retain distinct mean values across RFM features in original units.

---

## 1. Introduction

Customer segmentation via RFM (Recency, Frequency, Monetary) features is among the most widely deployed applications of unsupervised learning in commercial analytics [1, 21]. RFM encodes three behaviorally meaningful dimensions: elapsed time since a customer's last transaction (Recency), total transaction count in a reference period (Frequency), and cumulative spending (Monetary). Partitioning customers along these axes exposes qualitatively distinct behavioral profiles and directly informs retention campaigns, pricing strategies, and lifetime value models.

The operational appeal of RFM lies in its interpretability: cluster descriptions stated in original units ("customers spending $5,000-$10,000 annually with weekly purchase frequency") are immediately actionable. This interpretability depends on using features in their natural scales. Yet natural scales make RFM pathological for Euclidean distance. In the OR2 dataset examined in this paper, monetary values range up to $349,164, creating a monetary scale ratio of 936x over recency, and contributing more than 99.99% of squared Euclidean distances. Under such conditions, Euclidean-based clustering partitions customers by spending magnitude alone, ignoring recency and frequency entirely, producing what we call *monopolized clustering*: configurations where one or two clusters contain more than 96% of all customers.

The standard recommendation is feature standardization, which restores equal per-feature influence. However, standardization transforms cluster descriptions into abstract z-score units uninterpretable to marketing teams, introduces an arbitrary preprocessing choice that demonstrably affects clustering outcomes [10], and conflates statistical variance with business importance. A 10% deviation in purchase frequency may signal very different customer behavior than a 10% deviation in recency, yet standardization treats both equally. These concerns motivate investigating distance metric alternatives that achieve scale invariance without modifying the feature values themselves.

This paper presents an empirical investigation of CURE (Clustering Using Representatives) [2] and its large-scale variant on unstandardized RFM data, with three coordinated modifications to address monopolization. Our contributions are:

1. An empirical demonstration that Pearson correlation distance, substituted for Euclidean within an otherwise unmodified CURE algorithm, eliminates monopolized clustering in 55-82% of previously failing configurations across four real-world datasets, while preserving original-unit cluster descriptions.

2. An adaptation of centroid-proximity sampling to the sampling stage of CURE's large-scale variant (which we call ScalableCURE as a shorthand for the sampling-based algorithm described in [2]; the paper itself does not assign this name), drawing on the trimmed-core sampling principle from BIRCH [13].

3. A distributed trimming mechanism for MapReduce CURE, parameterized by an *outlier fraction* $f$, which extends the trimmed k-means principle [17] to the distributed CURE setting where per-mapper local filtering replaces global trimming.

4. A suite of evaluation metrics adapted for Pearson distance space, with explicit acknowledgment of their conceptual limitations relative to Euclidean-geometry assumptions.

5. Cluster profile analysis demonstrating that Pearson-based segments, despite using correlation rather than magnitude for assignment, yield cluster descriptions with distinct and interpretable mean values across all three RFM features.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 provides background on CURE, the scalable variant, MapReduce, and distance metrics. Section 4 formalizes the monopolized clustering problem. Section 5 presents our methodology. Section 6 describes the experimental evaluation. Section 7 discusses implications and limitations. Section 8 concludes.

---

## 2. Related Work

**Scale-Invariant Clustering.** The sensitivity of distance-based clustering to feature scales has been studied extensively. Milligan and Cooper [10] demonstrated that different standardization schemes yield substantially different clustering outcomes with no universally optimal choice, motivating the search for preprocessing-free alternatives. Huang et al. [11] proposed weighted k-means, which iteratively updates feature weights and cluster centroids, but still operates in absolute scale space and requires initialization of weights. Mahalanobis distance decorrelates features and normalizes by covariance, but requires full covariance matrix estimation. Cosine similarity, standard in high-dimensional text applications, closely related to Pearson correlation (they are equivalent when vectors are mean-centered), has not been systematically applied to CURE or RFM clustering. Gower's coefficient [12] handles heterogeneous feature types but normalizes each feature independently to [0,1], which is a form of standardization. To our knowledge, no prior work systematically investigates Pearson correlation distance within a CURE hierarchical clustering framework for multi-scale RFM data.

**Outlier-Resistant Clustering.** Garcia-Escudero et al. [17] introduced the trimmed clustering framework, of which trimmed k-means is a prominent instance: a fixed fraction of points most distant from cluster centers is excluded from centroid computation at each iteration, providing robustness to outliers. TCLUST [17] generalizes this to elliptical clusters. Our distributed outlier fraction parameter is conceptually a distributed analog of this trimming principle applied per MapReduce partition, and we position it explicitly as such in Section 5. Centroid-proximity sampling draws on the dense-core summarization principle used in BIRCH [13] and in CLARA's sampling strategy, adapted here to the initial sampling stage of ScalableCURE. LOF [15] and Isolation Forest [16] detect outliers as a preprocessing step, requiring separate threshold specification and potentially discarding valid data; our integrated approach excludes points from clustering while retaining them for final assignment.

**CURE and Variants.** CURE [2] introduced multi-representative cluster encoding with shrinking, enabling non-spherical cluster discovery with outlier resistance. The large-scale variant described in [2] employs random sampling and partitioning; we observe in our implementation that this default sampling is uniform random, providing no outlier resistance. Distributed CURE implementations have been proposed using MapReduce [4] but without outlier-aware sampling or investigation of metric choice. BIRCH [13] constructs a compact CF-tree summary for large datasets but targets spherical micro-clusters. ROCK [14] extends CURE to categorical data but does not address scale imbalance.

**Clustering Evaluation.** Silhouette Score [5], Davies-Bouldin Index [6], and Dunn Index [7] are defined in terms of pairwise distances, making them formally extensible to any dissimilarity function. Van Craenendonck and Blockeel [19] showed that metric choice significantly impacts internal validity indices. Eisen et al. [20] applied correlation-based Silhouette in genomic clustering where relative expression patterns matter more than absolute levels; the analogy to RFM behavioral patterns motivates our adaptation. We note that the Calinski-Harabasz index [8] relies on variance decomposition, which has a clean geometric interpretation under Euclidean distance but less so under Pearson distance; we adapt it pragmatically and discuss the conceptual limitations.

**RFM Segmentation.** Classical RFM scoring assigned discrete quintile ranks to each feature [22] before modern approaches adopted continuous-valued clustering. Most RFM clustering studies apply k-means or Gaussian mixture models with feature standardization [21, 23], sacrificing interpretability for scale invariance. No prior work, to our knowledge, proposes using a scale-invariant distance metric within hierarchical RFM clustering as an alternative to standardization.

---

## 3. Background

### 3.1 CURE Algorithm

CURE [2] represents each cluster with $c$ scattered representative points, each shrunk toward the cluster centroid by a factor $\alpha \in (0,1)$:
$$r^{(j)} \leftarrow \bar{c} + \alpha \cdot \bigl(p^{(j)} - \bar{c}\bigr)$$
where $p^{(j)}$ is the $j$-th selected scatter point and $\bar{c}$ is the cluster centroid. Inter-cluster distance is the minimum representative distance:
$$d(C_i, C_j) = \min_{r \in R_i,\, s \in R_j} d(r, s)$$

CURE maintains a KD-tree over active representatives and a min-heap ordered by closest-cluster distance. Starting from $n$ singleton clusters, it iteratively merges the closest pair until $k$ clusters remain, then assigns all $n$ points to their nearest representative.

The algorithm incurs $O(n^2 \log n)$ complexity due to KD-tree reconstruction after each merge. With $n - k$ merges, each requiring $O(m \log m)$ tree construction where $m$ decreases from $n$ to $k$, the total is $\sum_{m=k}^{n} O(m \log m) = O(n^2 \log n)$.

### 3.2 Scalable Variant of CURE

To reduce the $O(n^2 \log n)$ bottleneck, the large-scale variant described in [2] (which we call ScalableCURE as a shorthand) applies a five-stage pipeline:

1. **Sampling.** Draw a sample of size $s = \min(\max(50 \cdot n^{1/3}, 500), 15{,}000)$ from the dataset.
2. **Partitioning.** Divide the sample into $p$ equal partitions.
3. **Partial clustering.** Apply CURE independently to each partition, reducing to $\lfloor |P_j| / 3 \rfloor$ sub-clusters.
4. **Final clustering.** Apply CURE to all partial-cluster representatives to produce $k$ final clusters.
5. **Labeling.** Assign all $n$ original points to their nearest final representative via KD-tree.

The total complexity is $O(n \log n)$ since the sampling sort dominates; CURE sub-calls operate on $O(n^{1/3})$-sized inputs.

### 3.3 MapReduce CURE

We implement CURE on Apache Hadoop via Hadoop Streaming. Data is pre-partitioned into $p=25$ HDFS splits. Each mapper receives one split, applies local clustering, and emits representative points; the reducer merges all representatives into $k$ final clusters; the labeler assigns all $n$ points by nearest-representative KD-tree lookup.

### 3.4 Distance Metrics

**Euclidean distance:** $d_E(\mathbf{x},\mathbf{y}) = \|\mathbf{x}-\mathbf{y}\|_2$. Scale-sensitive: if Monetary has range $\Delta_M$ and Recency has range $\Delta_R$, the Monetary contribution to $d_E^2$ is proportional to $\Delta_M^2$. When $\Delta_M / \Delta_R = 936$, the Monetary term contributes $936^2 / (936^2 + 1^2 + \ldots) \approx 100\%$ of the squared distance.

**Pearson correlation distance:**
$$d_P(\mathbf{x},\mathbf{y}) = 1 - \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sigma_x \sigma_y}$$
where $\bar{x}, \sigma_x$ denote the per-vector mean and standard deviation. $d_P$ is scale-invariant: $d_P(\lambda\mathbf{x}, \mu\mathbf{y}) = d_P(\mathbf{x},\mathbf{y})$ for any $\lambda, \mu > 0$. Its range is $[0,2]$, with $d_P = 0$ for perfectly correlated vectors. Importantly, $d_P$ measures *pattern similarity*, not *magnitude similarity*: two vectors $[10, 5, 100]$ and $[100, 50, 1000]$ have $d_P = 0$ despite a tenfold difference in absolute values. We discuss in Section 5.3 what this means for cluster interpretability.

**Note on metric properties.** Pearson correlation distance does not satisfy all metric axioms and can violate the triangle inequality. CURE's inter-cluster distance oracle requires only a dissimilarity function, not a formal metric, so this is not an algorithmic issue.

**Note on dimensionality.** Pearson correlation distance is most commonly applied in genomics and time series analysis, where each vector has hundreds to thousands of features [20], providing statistical stability. For 3-dimensional RFM vectors, the correlation is computed over only three data points per vector, making it sensitive to individual feature values. In practice, this sensitivity is partially mitigated by the extreme skewness of the Monetary feature: when $M \gg R, F$, the Monetary component dominates the per-vector mean and standard deviation, stabilizing the correlation. However, this stability is dataset-specific and may not generalize to RFM data where Monetary does not strongly dominate, or to configurations with balanced feature ranges. Practitioners should be aware that Pearson distance on low-dimensional vectors is a pragmatic choice rather than a statistically principled one.

### 3.5 Evaluation Metrics for Pearson Distance

We adapt three internal validity indices for Pearson distance. All are formally well-defined for any dissimilarity function, though their geometric interpretations differ from the Euclidean case.

**Silhouette Score (SC)** [5]. For point $i$ in cluster $C_k$: $a(i)$ is the mean $d_P$ to co-cluster members; $b(i)$ is the minimum mean $d_P$ to any other cluster. Then $s(i) = (b(i)-a(i))/\max(a(i),b(i))$ and SC $= n^{-1}\sum_i s(i) \in [-1,1]$. Higher is better. Computed via scikit-learn with `metric='correlation'`.

**Davies-Bouldin Index (DB)** [6]. Let $\sigma_k = |C_k|^{-1}\sum_{i \in C_k} d_P(i, \bar{c}_k)$ where $\bar{c}_k$ is the centroid. Then DB $= K^{-1}\sum_k \max_{\ell \neq k} (\sigma_k + \sigma_\ell)/d_P(\bar{c}_k, \bar{c}_\ell)$. Lower is better. Note that the centroid $\bar{c}_k$ is computed in Euclidean feature space; under Pearson distance this centroid is not necessarily the "center" of the cluster. We acknowledge this conceptual inconsistency and treat DB as a practical approximation.

**Calinski-Harabasz Index (CH)** [8]. We adapt by replacing squared Euclidean distances with squared Pearson distances in the between-cluster and within-cluster scatter terms. This yields a correlation-analogue of the variance ratio criterion rather than a strict variance decomposition, and its interpretation is accordingly less clean than in the Euclidean case. We report it with this caveat and weight SC and DB more heavily in our conclusions.

---

## 4. Problem Formulation

**Definition 1 (Feature Dominance).** Given dataset $D \subset \mathbb{R}^d$ with per-feature ranges $\Delta_j = \max_i x_i^{(j)} - \min_i x_i^{(j)}$, feature $j$ *dominates* under Euclidean distance if $\Delta_j^2 \gg \sum_{\ell \neq j} \Delta_\ell^2$. Quantitatively, we say feature $j$ is dominant when its share of total squared range exceeds 99%: $\Delta_j^2 / \sum_\ell \Delta_\ell^2 > 0.99$.

In all four datasets we study (Table 1 below), the Monetary feature is dominant under this definition, with contributions ranging from 99.9% to 100%.

**Definition 2 (Monopolized Clustering).** A $K$-partition $\{C_1, \ldots, C_K\}$ of $D$ is *monopolized* if the Top-2 concentration $\tau = (|C_{(1)}| + |C_{(2)}|)/n \geq 0.96$, where $|C_{(1)}| \geq |C_{(2)}| \geq \ldots$ are sorted cluster sizes.

The 96% threshold reflects a practical condition: at this concentration level, the remaining $K-2$ clusters together contain fewer than 4% of data. For $K=4$, each minor cluster holds on average less than 2% of customers, making it commercially negligible for any targeted campaign. We acknowledge that this threshold is empirically motivated rather than statistically derived; Definitions at 90% or 95% yield qualitatively similar experimental conclusions, as we discuss in Section 7.2.

**Problem Statement.** Given an unstandardized RFM dataset $D$, produce a $K$-clustering that is (i) non-monopolized ($\tau < 0.96$); (ii) internally coherent (high SC, low DB under Pearson distance); (iii) interpretable in original feature units; and (iv) scalable to $n > 10^6$ within practical runtimes. The key structural constraint is that conditions (i) through (iii) jointly rule out both Euclidean distance on raw features (violates i) and standardization (violates iii).

---

## 5. Methodology

### 5.1 Pearson Distance in CURE

Substituting $d_P$ for $d_E$ in CURE requires modifying the inter-cluster distance oracle, the representative scatter selection, and the final labeling step. The representative shrinking operation $r \leftarrow \bar{c} + \alpha(r - \bar{c})$ remains in feature space, as it describes geometric interpolation in the original coordinate system.

**KD-tree approximation.** KD-trees partition space using axis-aligned splits optimized for Euclidean geometry. Using a KD-tree with Pearson distance may return incorrect nearest neighbors if the tree's Euclidean candidate set does not include the true Pearson-nearest representative. Our implementation addresses this by using the KD-tree as a filter to identify candidate neighbors, then computing exact $d_P$ for all candidates to select the true nearest representative. The approximation is exact for labeling (guaranteed nearest representative by $d_P$), at the cost of potentially examining more candidates than an exact Pearson tree would require. We do not have theoretical bounds on the candidate set overhead; this is a known limitation we flag for future work.

**Centroid mismatch.** The cluster centroid used in shrinking is the Euclidean mean of member points. This centroid is not the "center" under $d_P$ in any formal sense, since $d_P$ does not correspond to a vector space structure. In practice, the Euclidean centroid serves as a reasonable representative of the cluster's location for shrinking purposes, and the $\alpha$ parameter controls the degree to which representatives converge toward it. The potential inconsistency between Euclidean centroid and Pearson cluster structure is a known limitation.

### 5.2 Centroid-Proximity Sampling for ScalableCURE

The default sampling stage of ScalableCURE selects $s$ points uniformly at random, giving outliers equal probability of inclusion. We replace this with proximity-to-centroid selection, drawing on the dense-core sampling principle used in BIRCH [13] and CLARA:

**Algorithm 1: Centroid-Proximity Sampling**
```
Input:  data D (n points), sample size s, distance function delta
1. Compute centroid: c = mean(D)  [in feature space]
2. For each x in D: dist(x) = delta(x, c)
3. Return the s points with smallest dist values
```

This adaptation was motivated by the observation that outlier points, which are disproportionately responsible for monopolization under Euclidean distance, tend to be spatially extreme relative to the feature-space centroid. Points excluded from sampling still receive cluster assignments in the labeling stage; no data is discarded.

The complexity addition is $O(n \log n)$ for sorting, which does not change the asymptotic complexity of ScalableCURE. For Pearson distance, centroid-proximity sampling is less effective because Pearson already neutralizes scale dominance; its primary benefit is for Euclidean-based configurations.

### 5.3 Distributed Trimming via Outlier Fraction

In the MapReduce setting, each mapper processes a local partition $P_j$ of $m \approx n/p$ points. We introduce the outlier fraction $f \in [0,1]$, controlling what proportion of each partition is excluded from local clustering:

**Algorithm 2: Mapper with Outlier Fraction**
```
Input:  partition P (m points), f, (k_local, c, alpha, delta)
1. Compute local centroid c_local = mean(P)  [in feature space]
2. Sort P by delta(x, c_local) ascending
3. Retain Q = P[1 : ceil((1-f)*m)]  // closest (1-f) fraction
4. Run CURE(Q, k_local, c, alpha, delta)
5. Emit (cluster_id, representative) for each output representative
```

This is a distributed analog of the trimmed clustering framework [17]: rather than trimming globally based on distance to cluster centers (as in trimmed k-means), we trim locally per partition based on distance to the partition centroid. The per-partition (local) centroid is used rather than a global centroid because computing the latter requires an additional MapReduce pass; local trimming is a practical approximation. We note the resulting inter-partition inconsistency: a point that is central globally may be peripheral in its assigned partition, and vice versa. This inconsistency motivates the empirical $f$-sweep rather than a single fixed value.

Points excluded in Step 3 still receive cluster assignments from the labeler stage. The reducer and labeler are unchanged from standard MapReduce CURE.

### 5.4 Cluster Interpretability Under Pearson Distance

A critical question raised by Pearson's scale invariance is whether the resulting clusters remain meaningful in business terms. Since $d_P([10,5,100], [100,50,1000]) = 0$, customers with proportional RFM vectors receive identical cluster assignments regardless of absolute spending levels. One might fear that Pearson clusters would mix low-value and high-value customers indiscriminately, destroying value segmentation.

Empirically, this concern is not supported by our results. Table 4 (Section 6.5) shows that for the OR2 dataset, the four Pearson-based clusters have clearly distinct mean profiles across all three features, including Monetary ($\mu_M$ ranging from $492 to $7,010). For the 3A dataset (Table 5), clusters separate primarily by Recency ($\mu_R$ from 3.7 to 18.0 days), with meaningful Monetary variation ($\mu_M$ from $119,001 to $142,241). The reason proportional mixing does not dominate in practice is that truly proportional vectors (customers with identical behavioral ratios across all three features but different absolute scales) are statistically rare in real RFM data; empirical RFM distributions are highly skewed (skewness 1.28-23.98 across features and datasets, Table 1), making exact proportionality uncommon. The Pearson-based clusters that emerge reflect behavioral archetypes (e.g., "recent high-frequency high-spenders" vs. "dormant low-frequency low-spenders") that are both statistically coherent and commercially actionable.

We nonetheless acknowledge that Pearson distance may be unsuitable for applications where the primary segmentation objective is pure value tiering (e.g., separating customers by absolute spending quartile). For such applications, Euclidean distance with appropriate standardization or log-transform remains preferable.

---

## 6. Experimental Evaluation

### 6.1 Datasets

**Table 1: Dataset characteristics (all values in original unstandardized units).**

| Dataset | $n$ | R: mean (std, skew) | F: mean (std, skew) | M: mean (std, skew) | M dominance in $d_E^2$ |
|---------|-----|--------------------|--------------------|--------------------|-----------------------|
| Dunnhumby | 2,500 | 25.6 (62.8, 5.57) [0-657] | 110.6 (115.6, 3.37) [1-1300] | 3,223 (3,348, 2.41) [8-38K] | 99.9% |
| OR2 | 4,314 | 90.3 (96.9, 1.28) [0-373] | 4.5 (8.2, 10.54) [1-205] | 2,047 (8,912, 23.98) [0-349K] | ~100% |
| 3A | 99,996 | 8.8 (9.3, 2.00) [0-120] | 102.4 (10.1, 0.12) [62-202] | 131,062 (20,058, 0.90) [60K-483K] | ~100% |
| GFR | 1,283,707 | 217.5 (194.3, 1.10) [0-807] | 3.5 (2.8, 1.89) [1-35] | 574 (1,108, 4.27) [0-22K] | 99.9% |

All features exhibit positive skewness, consistent with the long-tailed spending distributions typical of retail e-commerce. Monetary dominates squared Euclidean distances by a factor of at least 99.9% in all datasets, confirming that Euclidean clustering on raw features reduces to a pure monetary-value partition.

**Experimental configuration.** All experiments use raw, unstandardized features and $K=4$ clusters. Hierarchical experiments sweep the shrinking factor $\alpha \in \{0.20, 0.25, \ldots, 0.70\}$ (11 values). Distributed experiments fix $\alpha=0.5$ and sweep $f \in \{0.0, 0.1, \ldots, 1.0\}$ (11 values). We set the representative count $c=10$ for Dunnhumby and OR2, $c=5$ for 3A and GFR, reflecting the trade-off between cluster shape fidelity and computational cost. Distributed experiments use 25 HDFS input splits on a 4-node Hadoop cluster (1 NameNode, 3 DataNodes).

**Reproducibility note.** All hierarchical experiments are deterministic given fixed seeds. Distributed experiments depend on HDFS partition assignment, which is deterministic given the same input file ordering. Each configuration is run once; we report single-run results and acknowledge that multi-run stability analysis would strengthen these findings.

### 6.2 Effect of Distance Metric on Monopolization

**Table 2: Monopolization rates under Euclidean vs. Pearson distance (ScalableCURE with random sampling, $\alpha$ swept).**

| Dataset | Euclidean monopolized / total | Pearson monopolized / total | Reduction (pp) |
|---------|------------------------------|----------------------------|---------------|
| Dunnhumby | 8/11 (73%) | 0/11 (0%) | 73 |
| OR2 | 6/11 (55%) | 0/11 (0%) | 55 |
| 3A | 9/11 (82%) | 1/11 (9%) | 73 |
| GFR | 11/11 (100%) | 2/11 (18%) | 82 |

Pearson distance eliminates monopolized clustering entirely in Dunnhumby and OR2, and reduces it from near-total to marginal levels in 3A and GFR. For GFR under Euclidean distance, the monetary dominance is absolute: the M/R scale ratio of 27 still contributes 99.9% of squared Euclidean distances, and no $\alpha$ value produces a balanced clustering. Pearson normalizes each per-vector dimension before computing correlations, restoring all three features to equal influence on cluster assignments.

The two remaining monopolized Pearson configurations for 3A occur at the smallest $\alpha$ values (0.20 and 0.25), where aggressive shrinking collapses representatives so close to centroids that cluster separation becomes unreliable. This suggests an interaction between $\alpha$ and Pearson distance that warrants further study.

### 6.3 Effect of Centroid-Proximity Sampling

Centroid-proximity sampling, as an adaptation of trimmed-core sampling to ScalableCURE's initialization stage, provides meaningful improvement for Euclidean distance but marginal benefit for Pearson:

- **Euclidean:** Eliminates monopolization in 2/6 OR2 configurations, 3/8 Dunnhumby configurations, 1/9 3A configurations. Provides no benefit for GFR (all 11 remain monopolized).
- **Pearson:** Reduces $\tau$ by 3-8 percentage points in already-balanced configurations. Does not rescue the two remaining monopolized configurations.

The differential effect is consistent with the mechanism: centroid-proximity sampling suppresses outliers that are spatially extreme in the high-magnitude Monetary dimension. Pearson distance already normalizes for this, reducing the marginal value of pre-filtering. For GFR under Euclidean distance, the scale ratio is too large for sampling-based filtering to overcome; the cluster merge algorithm still operates on Euclidean distances, and any sample drawn from a Monetarily skewed population will retain sufficient monetary variance to monopolize.

### 6.4 Outlier Fraction Sweep in Distributed CURE

Figure 1 (see `distributed_sweep_f_vs_top2.png`) plots the Top-2 concentration $\tau(f)$ for all four datasets. Table 3 summarizes the results.

**Table 3: Distributed CURE outlier fraction sweep ($K=4$, $\alpha=0.5$, Pearson distance). Monopolized (MONO) = $\tau \geq 0.96$. OR2 is included to show balance improvement even from a non-monopolized baseline.**

| Dataset | $\tau(f{=}0)$ | Monopolized? | Optimal $f$ | $\tau_{\min}$ | $\Delta\tau$ | Cluster sizes at optimal $f$ |
|---------|--------------|-------------|------------|--------------|-------------|------------------------------|
| 3A | 99.9% | Yes | 0.8 | 55.8% | -44.1 pp | 30,957 / 24,867 / 22,740 / 21,432 |
| Dunnhumby | 97.0% | Yes | 0.8 | 54.1% | -42.9 pp | 695 / 658 / 620 / 527 |
| GFR | 100.0% | Yes | 0.7 | 71.5% | -28.5 pp | 547K / 371K / 204K / 162K |
| OR2 | 86.8% | No | 0.6 | 73.2% | -13.6 pp | 2,361 / 795 / 595 / 563 |

At $f=0$ (no trimming), three of four datasets are monopolized. The $\tau(f)$ curves share a characteristic concave shape: monotonically decreasing improvement from $f=0$ to a dataset-specific optimum, followed by quality degradation as excessive filtering removes legitimate minority-cluster members. The optimal $f$ values vary: $f=0.6$ for OR2 (smaller, less skewed), $f=0.7$ for GFR (very large, moderate skew), and $f=0.8$ for 3A and Dunnhumby (extreme initial monopolization requiring aggressive filtering).

The cluster size distributions at optimal $f$ are qualitatively superior: 3A moves from [50,751 / 49,107 / 136 / 2] at $f=0$ to a well-balanced four-way split [30,957 / 24,867 / 22,740 / 21,432] at $f=0.8$, in which all four clusters are commercially viable. GFR at optimal $f=0.7$ yields [547K / 371K / 204K / 162K], where the largest cluster accounts for 42.6% of data -- not uniform, but a significant improvement over 100% concentration at $f=0$.

We note that OR2 at $f=0$ (τ=86.8%) is not monopolized under Definition 2 and is included for completeness as a case of balance improvement rather than monopolization resolution. The distributed outlier fraction mechanism thus serves two distinct purposes: resolving monopolized clusterings (3A, Dunnhumby, GFR) and improving general cluster balance (OR2).

### 6.5 Cluster Profile Interpretability

**Table 4: OR2 cluster profiles at optimal $f=0.6$ (Pearson, distributed, $K=4$).**

| Cluster | $n$ | Mean R (days) | Mean F (orders) | Mean M ($) | Interpretation |
|---------|-----|--------------|----------------|-----------|---------------|
| C3 | 795 | 8.7 | 12.4 | 7,010 | Recent high-frequency high-spenders (VIP) |
| C2 | 563 | 27.6 | 5.2 | 2,222 | Recent moderate customers |
| C0 | 595 | 46.0 | 3.5 | 1,423 | Occasional mid-recency customers |
| C1 | 2,361 | 143.9 | 1.8 | 492 | Dormant low-value customers (churn risk) |

**Table 5: 3A cluster profiles at optimal $f=0.8$ (Pearson, distributed, $K=4$).**

| Cluster | $n$ | Mean R (days) | Mean F (orders) | Mean M ($) | Interpretation |
|---------|-----|--------------|----------------|-----------|---------------|
| C3 | 30,957 | 3.7 | 103.2 | 119,001 | Most recent, high-frequency |
| C2 | 21,432 | 5.8 | 103.3 | 130,149 | Recent, very high-frequency |
| C1 | 22,740 | 8.5 | 102.7 | 136,116 | Moderate recency, high spenders |
| C0 | 24,867 | 18.0 | 100.2 | 142,241 | Least recent, highest spenders |

For OR2 (Table 4), the four clusters span a 14x range in mean monetary value ($492 to $7,010), an 8x range in mean recency (8.7 to 143.9 days), and a 7x range in mean frequency (1.8 to 12.4 orders). These profiles support clear business interpretations across all three dimensions.

For 3A (Table 5), the picture is more nuanced. Recency is the primary differentiator (R from 3.7 to 18.0 days, a 5x ratio), while Frequency variation is minimal (100.2-103.3, a 3% range) and Monetary variation is 19% ($119K to $142K). In this dataset, which has high-value B2B-like characteristics with a narrow Monetary range, Pearson clustering effectively produces recency-based segments with modest monetary differences. Whether the Monetary differences ($23K range in absolute terms, 19% in relative terms) are actionable depends on the business context. The interpretability claim thus holds most strongly for datasets with meaningful variation across multiple features (OR2), and should be understood more narrowly as recency-based segmentation for datasets like 3A where Monetary variation is compressed.

### 6.6 Scalability and Runtime

**Table 6: Observed runtimes ($K=4$, $\alpha=0.5$, Pearson distance).**

| Dataset | Hierarchical (best config) | Distributed (optimal $f$) | Ratio |
|---------|---------------------------|--------------------------|-------|
| Dunnhumby (2.5K) | 2.1 s | 7.3 s | 0.3x |
| OR2 (4.3K) | 3.2 s | 8.1 s | 0.4x |
| 3A (100K) | 12.4 min | 8.2 min | 1.5x |
| GFR (1.28M) | OOM / >120 min | 45 min | infeasible |

The crossover in favor of distributed processing occurs between 10K and 100K points. For small datasets, distributed overhead exceeds the computation cost. For 3A, distributed processing is 1.5x faster and achieves better cluster balance. For GFR, hierarchical approaches fail due to memory exhaustion; distributed processing is the only viable option.

### 6.7 Pearson-Adapted Metric Validation

Across the 44 distributed sweep configurations (4 datasets x 11 $f$ values), Pearson rank-order correlations between metrics and $\tau$ are:

- SC: $r = -0.73$ ($p < 0.001$) -- higher SC corresponds to lower $\tau$ (better balance)
- DB: $r = +0.68$ ($p < 0.001$) -- lower DB corresponds to lower $\tau$
- CH: $r = -0.41$ ($p < 0.01$) -- moderate correlation, consistent with known CH limitations

Using SC $< 0.2$ as a monopolization flag: sensitivity 92% (11/12 monopolized configurations correctly identified), specificity 88% (28/32 balanced configurations correctly cleared). This is adequate for practical quality gating.

Importantly, Euclidean-based SC scores are misleading on this data: monopolized configurations with Euclidean distance yield SC $\in [0.6, 0.9]$, because the dominant Monetary feature creates genuinely compact clusters in monetary space despite their behavioral heterogeneity. Pearson-based SC correctly assigns near-zero or negative scores to these degenerate configurations.

---

## 7. Discussion

### 7.1 On the Nature of Monopolized Clustering

Monopolized clustering is distinct from ordinary low-quality clustering in that the algorithm output is structurally invalid for the stated purpose: a $K=4$ segmentation with $\tau = 0.99$ effectively delivers one segment. Standard internal metrics do not detect this because scale dominance creates compact Euclidean clusters that score well. The Top-2 concentration $\tau$ is thus a necessary complement to internal validity measures, not a replacement.

The 96% threshold we use is practically motivated and somewhat arbitrary. Sensitivity analysis at 90% and 95% thresholds yields the same qualitative conclusions from Table 2 (Euclidean configurations fail at substantially higher rates than Pearson), with exact counts varying by 1-2 configurations per dataset. We recommend practitioners adopt a threshold consistent with their minimum actionable segment size.

### 7.2 On the Novelty of the Contributions

We wish to be clear about the nature of our contributions relative to prior work. Substituting Pearson distance for Euclidean in CURE is a straightforward engineering change, as CURE's distance oracle admits any dissimilarity function. The contribution is empirical: the demonstration that this substitution has a dramatic and consistent effect on monopolization across four diverse datasets is, to our knowledge, previously unreported. Similarly, centroid-proximity sampling is an adaptation of an established principle (BIRCH's dense-core sampling, CLARA's sampling bias) to ScalableCURE's initialization stage, not a fundamentally new method. The distributed outlier fraction mechanism is a distributed instance of the trimmed clustering principle [17], not an independent invention. The primary novelty of this work is the systematic empirical investigation of how these coordinated adaptations address a specific structural failure mode of hierarchical clustering on multi-scale data.

### 7.3 Limitations

**Baselines.** We do not compare against standardized Euclidean clustering (z-score, min-max, log-transform) in terms of final cluster quality. We note, however, that a log-transform of Monetary alone does not resolve the scale dominance problem: while $\log(1+M)$ reduces Monetary's share of squared Euclidean distances from ~100% to ~0.1%, the remaining features then dominate. For GFR, Recency takes over with 99.8% dominance after $\log(M)$; for OR2 it dominates at 76.9%; for 3A and Dunnhumby, Frequency dominates at 57-80%. Any single-feature transformation only transfers dominance to another feature. The only feature-space transformations that guarantee equal feature contributions are global normalizations (z-score, min-max), which is precisely what this paper argues against for interpretability reasons. Pearson distance differs from all feature-space transformations in that it normalizes per-vector-pair rather than per-feature-globally, making it scale-invariant by construction without modifying feature values. This theoretical argument does not substitute for an empirical comparison, which remains a direction for future work.

**Algorithm breadth.** Our investigation is limited to CURE variants. The monopolization problem likely affects other hierarchical methods under Euclidean distance on multi-scale data, but we do not verify this.

**Centroid sampling and tail coverage.** Algorithm 1 excludes points furthest from the feature-space centroid, which under Euclidean distance corresponds primarily to high-Monetary customers. In RFM segmentation, high-Monetary customers are often the most commercially significant. Excluding them from the clustering sample means cluster boundaries are shaped by the majority (lower-Monetary) customer population, with potential misassignment of VIP customers at labeling time. This is a deliberate trade-off -- removing high-Monetary outliers reduces their distorting effect on Euclidean-based clusters -- but it differs from BIRCH's CF-tree approach, which summarizes all points including tails. The implications for VIP customer segment quality merit separate investigation.

**Statistical stability.** Each configuration is run once. Algorithms using random sampling (ScalableCURE's uniform random mode, HDFS partition assignment) may show run-to-run variation. Multiple runs with standard deviation reporting would strengthen our conclusions.

**K sensitivity.** We fix $K=4$ throughout. Monopolization may behave differently at $K=3$ or $K=5$; the interaction between $K$ and the optimal $f$ is uncharacterized.

**Metric approximation cost.** SC and DB have $O(n^2)$ complexity for pairwise distance computation. We use sampling-based approximations for large datasets but do not systematically characterize approximation error.

**Cosine distance.** Cosine similarity is closely related to Pearson correlation (equivalent for mean-centered vectors) and would be a natural alternative comparison. We do not evaluate it here.

---

## 8. Conclusion

We investigated monopolized clustering as a structural failure mode of Euclidean-based CURE on unstandardized multi-scale RFM data, and demonstrated that three coordinated adaptations substantially reduce its prevalence: Pearson distance substitution, centroid-proximity sampling adapted from dense-core sampling methods, and distributed per-partition trimming via the outlier fraction parameter. Across four datasets spanning three orders of magnitude in size, Pearson distance eliminates monopolization in 55-82% of previously failing configurations while yielding cluster profiles with distinct and interpretable mean values in all three original RFM features.

The broader implication is that scale-invariant distance metrics offer a principled alternative to the standardization paradigm for multi-scale clustering: rather than transforming features into an abstract normalized space, the distance function itself is made scale-invariant, preserving original-unit cluster semantics. This distinction is consequential in business applications where cluster descriptions must communicate to domain experts in physical units.

Directions for future work include adaptive $f$ selection based on per-partition density statistics, systematic stability analysis across multiple random seeds, evaluation against standardized Euclidean baselines, and theoretical characterization of monopolization probability as a function of the feature scale ratio.

---

## References

[1] Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing Company.

[2] Guha, S., Rastogi, R., & Shim, K. (1998). CURE: An Efficient Clustering Algorithm for Large Databases. *Proc. ACM SIGMOD*, pp. 73-84.

[3] Guha, S., Rastogi, R., & Shim, K. (2001). Cure: An Efficient Clustering Algorithm for Large Databases. *Information Systems*, 26(1), 35-58.

[4] Dean, J., & Ghemawat, S. (2004). MapReduce: Simplified Data Processing on Large Clusters. *Proc. OSDI*, pp. 137-150.

[5] Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

[6] Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. *IEEE Trans. PAMI*, 1(2), 224-227.

[7] Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. *Journal of Cybernetics*, 3(3), 32-57.

[8] Calinski, T., & Harabasz, J. (1974). A Dendrite Method for Cluster Analysis. *Communications in Statistics*, 3(1), 1-27.

[9] Bentley, J. L. (1975). Multidimensional Binary Search Trees Used for Associative Searching. *Commun. ACM*, 18(9), 509-517.

[10] Milligan, G. W., & Cooper, M. C. (1988). A Study of Standardization of Variables in Cluster Analysis. *Journal of Classification*, 5(2), 181-204.

[11] Huang, J. Z., Ng, M. K., Rong, H., & Li, Z. (2005). Automated Variable Weighting in k-Means Type Clustering. *IEEE Trans. PAMI*, 27(5), 657-668.

[12] Gower, J. C. (1971). A General Coefficient of Similarity and Some of Its Properties. *Biometrics*, 27(4), 857-871.

[13] Zhang, T., Ramakrishnan, R., & Livny, M. (1996). BIRCH: An Efficient Data Clustering Method for Very Large Databases. *Proc. ACM SIGMOD*, pp. 103-114.

[14] Guha, S., Rastogi, R., & Shim, K. (1999). ROCK: A Robust Clustering Algorithm for Categorical Attributes. *Proc. ICDE*, pp. 512-521.

[15] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *Proc. ACM SIGMOD*, pp. 93-104.

[16] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *Proc. ICDM*, pp. 413-422.

[17] Garcia-Escudero, L. A., Gordaliza, A., Matran, C., & Mayo-Iscar, A. (2008). A General Trimming Approach to Robust Cluster Analysis. *Annals of Statistics*, 36(3), 1324-1345.

[18] Fraley, C., & Raftery, A. E. (2002). Model-Based Clustering, Discriminant Analysis, and Density Estimation. *Journal of the American Statistical Association*, 97(458), 611-631.

[19] Van Craenendonck, T., & Blockeel, H. (2017). Using Internal Validity Measures to Compare Clustering Algorithms. *Proc. Benelearn*.

[20] Eisen, M. B., Spellman, P. T., Brown, P. O., & Botstein, D. (1998). Cluster Analysis and Display of Genome-Wide Expression Patterns. *PNAS*, 95(25), 14863-14868.

[21] Tsiptsis, K. K., & Chorianopoulos, A. (2009). *Data Mining Techniques in CRM: Inside Customer Segmentation*. Wiley.

[22] Hughes, A. M. (2005). *Strategic Database Marketing* (3rd ed.). McGraw-Hill.

[23] Jain, A. K. (2010). Data Clustering: 50 Years Beyond K-Means. *Pattern Recognition Letters*, 31(8), 651-666.

---

## Appendix: Pseudocode

**ScalableCURE with Centroid-Proximity Sampling**
```
Input:  D (n points), K, c, alpha, use_centroid_sampling, delta
s = clamp(50 * n^(1/3), 500, 15000)
if use_centroid_sampling:
    c_bar = mean(D)
    S = s points in D nearest to c_bar under delta
else:
    S = random sample of size s from D
for j = 1..p:
    P_j = S[(j-1)*m : j*m]
    R_j = CURE(P_j, floor(|P_j|/3), c, alpha, delta)
C* = CURE(union_j R_j, K, c, alpha, delta)
for each x in D:
    label(x) = argmin_k d_delta(x, nearest rep of C*_k)
```

**Distributed CURE Mapper**
```
Input:  partition P, f, k_local, c, alpha, delta
c_local = mean(P)
Sort P by delta(x, c_local) ascending
Q = P[1 : ceil((1-f)*|P|)]
Emit representatives of CURE(Q, k_local, c, alpha, delta)
```
