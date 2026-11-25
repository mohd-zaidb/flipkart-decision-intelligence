"""Part 3: Segmentation (K-means, hierarchical clustering) and dimensionality reduction (PCA)."""
import os
import numpy as np
from shared import load_records, sample_mean_sd, write_csv, write_results_txt

COHORT_NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
COHORT_DIST = [
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
]


def run_kmeans(X_std, k):
    """Run deterministic K-means with pinned initial centroids and no iteration cap."""
    n = X_std.shape[0]
    step = n // k
    init_indices = [i * step for i in range(k)]
    centroids = X_std[init_indices].copy()
    prev_assign = None
    passes = 0

    while True:
        # Squared Euclidean distance to each centroid: shape (n, k)
        diff = X_std[:, None, :] - centroids[None, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        assign = np.argmin(sq_dists, axis=1)
        passes += 1
        if prev_assign is not None and np.array_equal(assign, prev_assign):
            break
        prev_assign = assign
        new_centroids = centroids.copy()
        for c_idx in range(k):
            mask = assign == c_idx
            if np.any(mask):
                new_centroids[c_idx] = np.sum(X_std[mask], axis=0) / int(np.sum(mask))
        centroids = new_centroids

    # WCSS measured with final centroids in standardised space
    assigned_centroids = centroids[assign]
    wcss = float(np.sum((X_std - assigned_centroids) ** 2))
    return assign, centroids, wcss, passes


def agglomerative_clustering(linkage):
    """Run agglomerative clustering on the 6-cohort dissimilarity table."""
    clusters = [[i] for i in range(6)]
    merges = []
    heights = []

    for merge_no in range(1, 6):
        best_pair = None
        best_dist = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_dists = [COHORT_DIST[a][b] for a in clusters[i] for b in clusters[j]]
                if linkage == "single":
                    d = min(pair_dists)
                elif linkage == "complete":
                    d = max(pair_dists)
                else:
                    d = sum(pair_dists) / len(pair_dists)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

        i, j = best_pair
        c_i, c_j = clusters[i], clusters[j]
        if min(c_i) < min(c_j):
            c_a, c_b = c_i, c_j
        else:
            c_a, c_b = c_j, c_i

        name_a = "+".join(COHORT_NAMES[idx] for idx in sorted(c_a))
        name_b = "+".join(COHORT_NAMES[idx] for idx in sorted(c_b))
        merges.append([linkage, merge_no, name_a, name_b, best_dist])
        heights.append(best_dist)

        merged = sorted(c_a + c_b)
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(merged)

    gaps = [heights[idx + 1] - heights[idx] for idx in range(4)]
    max_gap_idx = max(range(4), key=lambda idx: gaps[idx])
    merges_below_cut = max_gap_idx + 1
    clusters_at_cut = 6 - merges_below_cut
    return merges, clusters_at_cut


def main():
    _, _, X_all, outcome, value = load_records()
    X_cont = X_all[:, :5]
    n = X_cont.shape[0]

    # Task 3.1: Standardise X1..X5 over all 800 rows, run K-means for K=2..6
    _, sd_cont = sample_mean_sd(X_cont, axis=0)
    mean_cont = np.sum(X_cont, axis=0) / n
    X_std = (X_cont - mean_cont) / sd_cont

    wcss_by_k = {}
    assign_k3 = None
    for k in range(2, 7):
        assign, _, wcss, _ = run_kmeans(X_std, k)
        wcss_by_k[k] = wcss
        if k == 3:
            assign_k3 = assign

    # Task 3.2: Elbow rule
    elbow_k = None
    for k in (3, 4, 5):
        drop_next = wcss_by_k[k] - wcss_by_k[k + 1]
        drop_prev = wcss_by_k[k - 1] - wcss_by_k[k]
        if drop_next < 0.5 * drop_prev:
            elbow_k = k
            break

    # Task 3.3: Profile K=3 segments ordered by ascending mean of X3 (index 2)
    cluster_stats = []
    for c_idx in range(3):
        mask = assign_k3 == c_idx
        size = int(np.sum(mask))
        means_orig = np.sum(X_cont[mask], axis=0) / size
        outcome_rate = float(np.sum(outcome[mask]) / size)
        mean_val = float(np.sum(value[mask]) / size)
        cluster_stats.append({
            "cluster_id": c_idx,
            "mask": mask,
            "size": size,
            "means": means_orig,
            "outcome_rate": outcome_rate,
            "mean_value": mean_val,
        })

    cluster_stats.sort(key=lambda d: float(d["means"][2]))
    segment_labels = ["A", "B", "C"]
    segment_rows = []
    segment_masks = {}
    segment_sizes = {}
    for label, stats in zip(segment_labels, cluster_stats):
        segment_masks[label] = stats["mask"]
        segment_sizes[label] = stats["size"]
        segment_rows.append([
            label,
            stats["size"],
            float(stats["means"][0]),
            float(stats["means"][1]),
            float(stats["means"][2]),
            float(stats["means"][3]),
            float(stats["means"][4]),
            stats["outcome_rate"],
            stats["mean_value"],
        ])
    write_csv(
        os.path.join("outputs", "part3_segments.csv"),
        ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
        segment_rows,
    )

    # Task 3.4: Hierarchical clustering
    all_merges = []
    clusters_at_cut = {}
    for linkage in ("single", "complete", "average"):
        merges, n_clusters = agglomerative_clustering(linkage)
        all_merges.extend(merges)
        clusters_at_cut[linkage] = n_clusters
    write_csv(
        os.path.join("outputs", "part3_hierarchical.csv"),
        ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
        all_merges,
    )

    # Task 3.5: PCA by the seven steps
    # Step 1 & 2: mean vector and centre
    pca_mean = np.sum(X_std, axis=0) / n
    X_c = X_std - pca_mean
    # Step 3: covariance matrix S = (1/(n-1)) X^T X
    S = (X_c.T @ X_c) / (n - 1)
    # Step 4: eigenvalues and eigenvectors via numpy.linalg.eigh
    eigvals, eigvecs = np.linalg.eigh(S)
    # Step 5: sort descending and apply sign convention
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for col in range(eigvecs.shape[1]):
        vec = eigvecs[:, col]
        max_idx = int(np.argmax(np.abs(vec)))
        if vec[max_idx] < 0:
            eigvecs[:, col] = -vec

    eigenvalue_sum = float(np.sum(eigvals))
    explained_ratios = [float(ev / eigenvalue_sum) for ev in eigvals]
    cumulative_ratios = []
    running = 0.0
    for er in explained_ratios:
        running += er
        cumulative_ratios.append(running)

    components_for_80pct = next(idx + 1 for idx, cr in enumerate(cumulative_ratios) if cr >= 0.8)

    # Step 6 & 7: top 2 eigenvectors and projection Z = X W
    W = eigvecs[:, :2]
    Z = X_c @ W

    mean_pc1_seg = {}
    for label in segment_labels:
        mask = segment_masks[label]
        mean_pc1_seg[label] = float(np.sum(Z[mask, 0]) / segment_sizes[label])

    results = [
        ("wcss_k2", wcss_by_k[2]),
        ("wcss_k3", wcss_by_k[3]),
        ("wcss_k4", wcss_by_k[4]),
        ("wcss_k5", wcss_by_k[5]),
        ("wcss_k6", wcss_by_k[6]),
        ("elbow_k", elbow_k),
        ("size_segment_A", segment_sizes["A"]),
        ("size_segment_B", segment_sizes["B"]),
        ("size_segment_C", segment_sizes["C"]),
        ("single_clusters_at_cut", clusters_at_cut["single"]),
        ("complete_clusters_at_cut", clusters_at_cut["complete"]),
        ("average_clusters_at_cut", clusters_at_cut["average"]),
        ("eigenvalue_sum", eigenvalue_sum),
    ]
    for i in range(5):
        results.append((f"eigenvalue_{i + 1}", float(eigvals[i])))
    for i in range(5):
        results.append((f"explained_ratio_{i + 1}", explained_ratios[i]))
    for i in range(5):
        results.append((f"cumulative_ratio_{i + 1}", cumulative_ratios[i]))
    results.append(("components_for_80pct", components_for_80pct))
    for i in range(5):
        results.append((f"pc1_loading_X{i + 1}", float(W[i, 0])))
    for i in range(5):
        results.append((f"pc2_loading_X{i + 1}", float(W[i, 1])))
    for r_idx in range(3):
        results.append((f"row{r_idx + 1}_pc1", float(Z[r_idx, 0])))
        results.append((f"row{r_idx + 1}_pc2", float(Z[r_idx, 1])))
    for label in segment_labels:
        results.append((f"mean_pc1_segment_{label}", mean_pc1_seg[label]))

    write_results_txt(os.path.join("outputs", "part3_results.txt"), results)


if __name__ == "__main__":
    main()
