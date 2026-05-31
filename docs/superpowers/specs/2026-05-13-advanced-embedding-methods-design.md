# Advanced Embedding & Clustering Methods — Design Spec

**Date:** 2026-05-13 (revised 2026-05-14 after taxonomy + sample data arrived)
**Project:** Big Data Final — DNA Sequence Clustering of 314 MAGs
**Status:** Design v2 approved by user; ready for implementation plan refresh

## Context

The project already has baseline embeddings (PCA, t-SNE, UMAP) in `02_embeddings.ipynb`, evaluated by Silhouette on k-means cluster assignments. The 4-mer feature matrix is 314 × 256. Two new label files have arrived since v1 of this spec:

- `data/spire_samples.csv` — `sample_id` for **all 314 MAGs** across 41 unique samples (1–27 MAGs per sample).
- `data/spire_taxonomy.csv` — full 7-level taxonomy (domain through species) plus `sample_id`, `completeness`, `contamination` for **126/314 MAGs (~40%)**.

This unlocks (a) evaluation against true taxonomy on the labeled subset, (b) semi-supervised UMAP using family labels with `-1` for the unlabeled 188, and (c) a full-dataset sample/host confound check.

## Goal

Add new dimensionality-reduction and clustering methods genuinely applicable to non-negative 4-mer count data on N=314 samples; tune UMAP; and evaluate every method against both discovered clusters (geometric separability) and true taxonomy (biological correctness). Detect any sample/host confound that would invalidate biological interpretation.

## Non-goals

- **Supervised methods other than semi-supervised UMAP.** Supervised t-SNE has no clean library support; building a classifier on embeddings drifts from clustering scope.
- **Methods redundant with the existing baseline.** The baseline `02_embeddings.ipynb` already has default-parameter t-SNE — Kobak's small-N tunings (PCA init + multi-scale perplexity, learning rate N/12) are within noise at N=314, so we drop them. Drop NMF n_components=2 (a 2-component NMF scatter is too coarse to read); keep NMF n_components=10 for clustering.
- **Out-of-sample / parametric embeddings.** No need to embed new MAGs yet.
- **Pairwise sequence alignment** (Needleman-Wunsch, Progressive MSA) — computationally infeasible on full genomes.
- **SVD / Funk SVD / VAE / dna2vec** — SVD ≈ PCA on centered data; Funk SVD is for sparse rating matrices; VAE will overfit 314 samples; dna2vec embeds k-mers not genomes and adds significant complexity.

## Organization

**New notebook:** `03_advanced_methods.ipynb`. Keeps the existing 492 KB `02_embeddings.ipynb` as the baseline reference and isolates new work for clarity.

**Inputs (consumed, not regenerated):**
- `data/kmer4_features.npy` — 314 × 256
- `data/kmer4_names.npy` — MAG IDs, row-aligned
- `data/metadata.csv` — assembly stats (used for coloring scatters by GC)
- `data/spire_samples.csv` — `sample_id` for all 314 MAGs
- `data/spire_taxonomy.csv` — taxonomy + QC for 126 MAGs

**Outputs:**
- `data/embeddings/{nmf_10d,autoencoder_2d,umap_tuned_2d,umap_semisup_2d,pca_2d,pca_10d}.npy`
- `data/clusters/{kmeans,agglomerative,hdbscan}_labels_*.npy`
- `data/silhouette_results.csv` (long format, includes both k-means Silhouette and taxonomic Silhouette at phylum/family/genus)
- `data/cluster_purity.csv` (per-method per-cluster majority-family purity + AMI)
- `data/sample_confound.csv` (per-method ARI vs sample partition)
- `data/umap_sweep_results.csv`
- `data/silhouette_advanced_comparison.png`
- `data/embeddings_advanced_grid_kmeans.png` (colored by k-means cluster)
- `data/embeddings_advanced_grid_phylum.png` (colored by phylum, gray=unlabeled)
- `data/embeddings_advanced_grid_gc.png` (colored by GC content)
- `data/dendrogram.png`
- `data/ari_matrix.png`

**Notebook section order:**
1. Imports, load features, load labels (samples + taxonomy), recompute PCA baseline (PCA-2d for the grid, PCA-10d for clustering). Encode family labels for semi-supervised UMAP (family → integer; `-1` for unlabeled).
2. NMF (10-d only; for clustering)
3. Autoencoder (2-d bottleneck)
4. UMAP hyperparameter sweep
5. Semi-supervised UMAP using family labels
6. Agglomerative clustering + dendrogram
7. HDBSCAN on best tuned UMAP
8. Comparison: k-means Silhouette, taxonomic Silhouette (phylum/family/genus), cluster purity, sample confound, scatter grids, ARI heatmap, writeup

## Per-method specifications

### NMF (Lecture 3)

- **Library:** `sklearn.decomposition.NMF`
- **Input scaling:** L1-normalize rows so each genome row sums to 1 (k-mer frequencies; corrects for genome-size variation).
- **Fit:** `n_components=10` only (no 2-component fit — would be too coarse for visualization).
- **Init:** `nndsvd` (deterministic).
- **`max_iter`:** 1000.
- **Use:** the 10-d representation feeds k-means and Silhouette (parallel to PCA-10d). No NMF scatter panel in the grid.
- **Interpretability output:** print top-5 k-mers per component (each component is a "k-mer signature").
- **Why this fits:** k-mer frequencies are non-negative by construction; NMF gives parts-based, interpretable components that PCA cannot.

### Autoencoder (Lecture 4)

- **Framework:** PyTorch.
- **Architecture:** `256 → 64 → 16 → 2 → 16 → 64 → 256`, ReLU hidden, linear output.
- **Loss:** MSE reconstruction.
- **Optimizer:** Adam, lr=1e-3.
- **Training:** batch_size=32, 200 epochs, no regularization.
- **Input scaling:** L1-normalized rows (same as NMF).
- **Reproducibility:** `torch.manual_seed(42)`, `numpy.random.seed(42)`.
- **Output:** 2-d bottleneck activations + train-loss curve.
- **Why this fits:** captures non-linear k-mer interactions that PCA/NMF cannot. Likely to underperform UMAP at N=314 — the comparison itself is informative (note in writeup).

### UMAP hyperparameter sweep (Lecture 5)

- **Grid:** `n_neighbors ∈ {5, 15, 30, 50}` × `min_dist ∈ {0.0, 0.1, 0.5}` → 12 fits.
- **Selection metric:** Silhouette of k-means(k=4) on each embedding.
- **Outputs:** `umap_tuned_2d.npy` (best), `umap_sweep_results.csv`.

### Semi-supervised UMAP (Lecture 5, label-aware)

- **Library:** `umap.UMAP(..., random_state=42).fit_transform(X, y=family_labels)`.
- **Labels:** integer family codes for the 126 labeled MAGs; `-1` for the 188 unlabeled MAGs (UMAP's native semi-supervised mode).
- **Hyperparameters:** use the best `n_neighbors` / `min_dist` from §UMAP sweep so the comparison isolates the effect of labels.
- **Why this fits:** directly tests H2 (sequence space recapitulates taxonomy). If family signal is in the 4-mer features, this embedding should sharpen family clusters and the unlabeled 188 MAGs should fall into the same regions as their nearest labeled neighbors.
- **Caveat:** evaluation on the same labels used during fit is circular. Report semi-sup UMAP's taxonomic Silhouette but flag this explicitly; primary fair comparison is against unsupervised methods.

### Agglomerative clustering (Lecture 7)

- **Libraries:** `scipy.cluster.hierarchy` + `sklearn.cluster.AgglomerativeClustering`.
- **Distance:** cosine on 4-mer frequencies (Lecture 7's q-gram cosine).
- **Linkage:** `average`.
- **Flat cut:** k ∈ {2..10} maximizing Silhouette on 4-mer cosine space.

### HDBSCAN

- **Library:** `hdbscan`.
- **Input:** the best tuned UMAP-2D from §UMAP sweep.
- **Params:** `min_cluster_size=10`, `min_samples=5`.
- **Noise:** label -1 plotted in gray; excluded from Silhouette; `n_noise / N` reported.

## Evaluation protocol

Every embedding gets the same evaluation treatment.

### Internal (geometric) evaluation — full dataset, 314 MAGs

For each embedding (PCA-10d, NMF-10d, AE-2d, UMAP-tuned-2d, UMAP-semisup-2d):
1. Run k-means with `k ∈ {2, 3, 4, 5, 6, 8, 10}`.
2. Compute Silhouette on the embedding space (euclidean).
3. Record best k and best score.

Agglomerative: Silhouette over k ∈ {2..10} on 4-mer cosine.
HDBSCAN: single score on discovered partition (noise excluded); `n_noise/N` reported.

### External (biological) evaluation — labeled subset, 126 MAGs

For each embedding (same five) and the agglomerative clusterer:
1. Compute Silhouette score on the labeled subset, with cluster labels = true taxonomy at level L, for L ∈ {phylum, family, genus}.
2. Report all three levels in `silhouette_results.csv`.

Treat semi-supervised UMAP's family-level taxonomic Silhouette as a sanity check rather than a fair comparison — it was fit with those labels.

### Cluster purity & AMI — labeled subset

For each method's best k-means partition, restricted to the 126 labeled MAGs:
- **Majority-family purity per cluster:** for each cluster, fraction of MAGs from its dominant family.
- **Overall purity:** sum of majority counts / total labeled count in clusters.
- **AMI (Adjusted Mutual Information):** between k-means labels and family labels.

Saved as `cluster_purity.csv` with columns `method, cluster_id, n_mags_labeled, dominant_family, purity, ami_overall`.

### Sample/host confound check — full dataset, 314 MAGs

For each method's best k-means partition (full 314), compute ARI between cluster labels and `sample_id` (41 groups).

**Interpretation:**
- ARI > 0.5 → strong sample/host confound; clusters partially reflect sample identity rather than biology.
- ARI ~ 0 → clusters independent of sample membership (good).
- High ARI is a *finding*, not a failure — but invalidates "these clusters are biological" without further analysis.

Saved as `sample_confound.csv` with columns `method, n_samples, sample_ari`.

## Output artifacts

### `silhouette_results.csv` (long format)

Columns: `method, clusterer, k, silhouette_internal, silhouette_phylum, silhouette_family, silhouette_genus, n_noise`

`silhouette_internal` is the k-means/agglomerative/HDBSCAN Silhouette in the embedding/feature space; the three taxonomic columns are computed only on the 126 labeled MAGs against true taxonomy at each level. Cells are NaN where not applicable.

### `cluster_purity.csv`

Columns: `method, cluster_id, n_mags_labeled, dominant_family, purity, ami_overall`. One row per (method, cluster) pair on the labeled subset.

### `sample_confound.csv`

Columns: `method, n_samples, sample_ari`. One row per method.

### `silhouette_advanced_comparison.png`

Grouped bar chart: x = method, y = Silhouette. Three bar groups per method: internal Silhouette, family-level taxonomic Silhouette, sample-ARI (the confound signal — high here is bad).

### `embeddings_advanced_grid_kmeans.png`

2×3 panel grid colored by k-means cluster at each method's best k. Panels:
- Row 1: PCA, NMF (via PCA-2d coords reused for layout — see note below), Autoencoder
- Row 2: UMAP-tuned, UMAP-semisup, UMAP + HDBSCAN

**Note:** NMF has no 2-component fit. The "NMF" panel uses PCA-2d coordinates for layout but with point colors from k-means clustering on NMF-10d, so the NMF *clustering* is shown even though its *layout* is borrowed. Title: "PCA-2d layout, NMF-10d clusters". Alternative: drop the NMF panel and make the grid 2×3 with: PCA / AE / UMAP-tuned / UMAP-semisup / UMAP-HDBSCAN / Agglomerative-on-PCA-layout. Default chosen here: borrowed-layout panel because it keeps NMF visible in the comparison.

### `embeddings_advanced_grid_phylum.png`

Same six panels colored by phylum (gray for unlabeled 188 MAGs). Visual check: do phyla separate in any embedding?

### `embeddings_advanced_grid_gc.png`

Same six panels colored by GC content (continuous viridis). Sanity check: is GC the dominant signal driving the structure?

### `dendrogram.png`

Agglomerative tree with red horizontal line at the chosen cut height.

### `ari_matrix.png`

Heatmap of Adjusted Rand Index between cluster assignments from each (embedding, clusterer) pair. Tells the reader whether methods agree on *which* MAGs are clustered together, independent of which method scores highest.

## Failure-mode handling

| Failure | Detection | Response |
|---|---|---|
| Autoencoder bottleneck collapse | std of either bottleneck dim < 0.01 | Retrain with seed=123; note in writeup |
| HDBSCAN labels everything noise | `n_noise == N` | Lower `min_cluster_size` to 5; if still all-noise, report as a finding |
| NMF non-convergence | sklearn warning | `max_iter=1000` from the start; bump to 2000 if still warns |
| Semi-sup UMAP fails (e.g., too few labels per family) | UMAP error or degenerate output | Merge rare families (< 3 MAGs) into "Other" before encoding; document the merging |
| Family label encoding mismatch | KeyError when joining taxonomy to features | Inner-join on `mag_id`; verify 126 rows align with rows of `X` by `mag_ids` order |

## Discussion expectations (markdown at end of notebook)

- Which method gave highest internal Silhouette, and which gave highest **family-level taxonomic Silhouette**? Where do they agree / disagree?
- Sample confound: what is the highest sample-ARI across methods? If > 0.5 for any method, discuss whether that method is picking up host/batch effects.
- Cluster purity: which method produces the most family-pure clusters? Are most clusters dominated by a single family, or are they mixed?
- ARI between methods: high agreement → robust signal; low agreement → fragility.
- HDBSCAN noise: do noise points correlate with sample, GC, or contig stats?
- NMF interpretability: do the top-k-mers per component look biologically meaningful (e.g., GC-rich signature in one component, AT-rich in another)?

## Stretch goals (time permitting)

1. **QC filter** — re-run comparison on completeness ≥ 50% and contamination ≤ 5% subset of the 126 labeled MAGs. Discuss whether QC filtering changes rankings.
2. **Day-of-sampling mapping** — if the 41 sample_ids can be mapped to days 0/1/7/30, test H1 (succession) by Silhouette / ARI against day.
3. **Parametric UMAP** for out-of-sample on other datasets (Devoto, Rampelli).

## Dependencies to add

Verify or add to `requirements.txt`:
- `scikit-learn` (already in)
- `umap-learn` (already in)
- `hdbscan` (new)
- `torch` (new)
- `scipy` (already in transitively)

Note: `openTSNE` is **removed** from the dependency list — Kobak-tuned t-SNE has been dropped from scope.
