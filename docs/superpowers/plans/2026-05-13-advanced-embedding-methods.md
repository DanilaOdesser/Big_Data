# Advanced Embedding & Clustering Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-05-13 (revised 2026-05-14 after taxonomy + sample data arrived)

**Goal:** Build `03_advanced_methods.ipynb` that adds NMF (10-d only), Autoencoder, Semi-supervised UMAP, Agglomerative clustering, and HDBSCAN, with a UMAP hyperparameter sweep. Evaluate every method against (a) internal Silhouette, (b) taxonomic Silhouette at phylum/family/genus on the 126-MAG labeled subset, (c) cluster taxonomic purity, and (d) sample/host confound ARI on the full 314 MAGs.

**Architecture:** One self-contained Jupyter notebook reading `data/kmer4_features.npy` (314 × 256), `data/spire_taxonomy.csv` (126 labeled), and `data/spire_samples.csv` (314 sample IDs). Writes 2D embeddings, cluster labels, a multi-axis Silhouette CSV, purity CSV, sample-confound CSV, three scatter grids (k-means / phylum / GC), dendrogram, and ARI heatmap. PCA baseline is recomputed inside this notebook so all comparisons use identically processed data.

**Tech Stack:** NumPy, pandas, matplotlib, scikit-learn, scipy, umap-learn, hdbscan, PyTorch

**Scope changes from v1:**
- **Dropped:** Kobak-tuned t-SNE (Kobak tunings don't move the needle at N=314; baseline t-SNE in `02_embeddings.ipynb` already covers L2).
- **Dropped:** NMF n_components=2 (2-component NMF scatter too coarse for visualization). NMF-10d kept for clustering.
- **Added:** Loading of taxonomy + sample labels (new Task 1b).
- **Added:** Semi-supervised UMAP (new Task 5b).
- **Added:** Taxonomic Silhouette at phylum/family/genus (extends Task 8).
- **Added:** Cluster taxonomic purity table + AMI (extends Task 8).
- **Added:** Sample/host confound ARI (extends Task 8).
- **Added:** Two additional scatter grid variants colored by phylum and by GC.
- **Removed dependency:** `openTSNE`.

**Notebook-project adaptation:** No git repo, no pytest. Each task implements one notebook section with inline `assert` sanity checks and a verification step listing expected files. No commit steps.

---

## File Structure

**Files to create:**
- `03_advanced_methods.ipynb` — the new notebook
- `data/embeddings/` — directory for new embedding arrays
- `data/clusters/` — directory for cluster label arrays
- `data/silhouette_results.csv` — long-format evaluation table (internal + phylum + family + genus)
- `data/cluster_purity.csv` — per-method per-cluster purity + AMI
- `data/sample_confound.csv` — per-method sample ARI
- `data/umap_sweep_results.csv` — UMAP hyperparameter grid scores
- `data/silhouette_advanced_comparison.png`
- `data/embeddings_advanced_grid_kmeans.png`
- `data/embeddings_advanced_grid_phylum.png`
- `data/embeddings_advanced_grid_gc.png`
- `data/dendrogram.png`
- `data/ari_matrix.png`

**Files to modify:**
- `requirements.txt` — add `hdbscan` and `torch` (do not add `openTSNE`)

**Files to read (not modify):**
- `data/kmer4_features.npy` — input feature matrix (314 × 256)
- `data/kmer4_names.npy` — MAG IDs, row-aligned
- `data/metadata.csv` — assembly stats (for GC coloring)
- `data/spire_taxonomy.csv` — taxonomy for 126 MAGs
- `data/spire_samples.csv` — sample_id for all 314 MAGs

---

## Task 0: Environment setup

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add two new dependencies**

Update `requirements.txt` to:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
umap-learn
tqdm
biopython
jupyter
hdbscan
torch
```

- [ ] **Step 2: Install new packages**

Run:
```bash
pip install hdbscan torch
```

Expected: successful install. If `hdbscan` fails to build, install build tools (`pip install --upgrade pip setuptools wheel cython` then retry). If `torch` is slow, use CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

- [ ] **Step 3: Verify imports**

```python
import hdbscan, torch
print(hdbscan.__version__, torch.__version__)
```

Expected: two version strings, no ImportError.

- [ ] **Step 4: Create output subdirectories**

```bash
mkdir -p /Users/danila/PycharmProjects/Big_Data/data/embeddings
mkdir -p /Users/danila/PycharmProjects/Big_Data/data/clusters
ls /Users/danila/PycharmProjects/Big_Data/data/embeddings /Users/danila/PycharmProjects/Big_Data/data/clusters
```

Expected: both directories listed, empty.

---

## Task 1: Notebook scaffold, load features and labels, recompute PCA baseline

**Files:**
- Create: `03_advanced_methods.ipynb`

This task creates the notebook with imports, a unified data-loading cell (features + sample_id + taxonomy), label encoding for semi-supervised UMAP, and a PCA baseline recompute. PCA is recomputed here (not loaded from `02_embeddings.ipynb`) so all downstream methods operate on identically scaled features.

- [ ] **Step 1: Add title markdown cell**

```markdown
# 03 — Advanced Embedding & Clustering Methods

Adds NMF-10d (for clustering), Autoencoder, UMAP hyperparameter sweep, Semi-supervised UMAP, Agglomerative clustering, and HDBSCAN. Evaluates against internal Silhouette and against true taxonomy at phylum/family/genus on the 126-MAG labeled subset. Checks sample/host confound on the full dataset.

Inputs:
- `data/kmer4_features.npy` (314 × 256, raw 4-mer counts)
- `data/spire_taxonomy.csv` (126 MAGs with full taxonomy)
- `data/spire_samples.csv` (sample_id for all 314 MAGs)
```

- [ ] **Step 2: Add imports cell**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize, LabelEncoder

import scipy.cluster.hierarchy as sch
import umap
import hdbscan
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("data")
EMB_DIR = DATA_DIR / "embeddings"
CLU_DIR = DATA_DIR / "clusters"
EMB_DIR.mkdir(exist_ok=True)
CLU_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 100
```

- [ ] **Step 3: Add feature-loading + normalization cell**

```python
X_counts = np.load(DATA_DIR / "kmer4_features.npy")
mag_ids = np.load(DATA_DIR / "kmer4_names.npy", allow_pickle=True)
metadata = pd.read_csv(DATA_DIR / "metadata.csv")

# L1-normalize rows: each MAG row sums to 1 (k-mer frequencies, corrects for genome size).
X = normalize(X_counts, norm="l1", axis=1)

assert X.shape == (314, 256), f"Expected (314, 256), got {X.shape}"
assert np.allclose(X.sum(axis=1), 1.0), "Rows should sum to 1 after L1 normalization"
assert len(mag_ids) == 314
print(f"Loaded X shape={X.shape}, mag_ids[0]={mag_ids[0]}")
```

- [ ] **Step 4: Add sample + taxonomy loading cell**

```python
samples_df = pd.read_csv(DATA_DIR / "spire_samples.csv")
taxonomy_df = pd.read_csv(DATA_DIR / "spire_taxonomy.csv")

# Some spire_samples.csv files have an odd leading token in the header line; normalize.
samples_df.columns = [c.strip().split()[-1] for c in samples_df.columns]
assert {"mag_id", "sample_id"}.issubset(samples_df.columns), f"samples columns: {samples_df.columns.tolist()}"

# Build row-aligned arrays indexed by the order of mag_ids
mag_id_list = [str(m) for m in mag_ids]
sample_lookup = dict(zip(samples_df["mag_id"], samples_df["sample_id"]))
sample_ids = np.array([sample_lookup.get(m, "UNKNOWN") for m in mag_id_list])
n_unknown_samples = int((sample_ids == "UNKNOWN").sum())
assert n_unknown_samples == 0, f"{n_unknown_samples} MAGs missing sample_id"
print(f"Samples: {len(set(sample_ids))} unique sample_ids across {len(sample_ids)} MAGs")

# Taxonomy is only available for ~126 MAGs. Build row-aligned arrays with NaN where unlabeled.
tax_lookup = taxonomy_df.set_index("mag_id")
tax_levels = ["phylum", "family", "genus"]
tax_arrays = {}
for level in tax_levels:
    tax_arrays[level] = np.array([
        tax_lookup.loc[m, level] if m in tax_lookup.index and pd.notna(tax_lookup.loc[m, level]) and tax_lookup.loc[m, level] != ""
        else None
        for m in mag_id_list
    ], dtype=object)

labeled_mask = np.array([tax_arrays["family"][i] is not None for i in range(314)])
print(f"Labeled MAGs: {labeled_mask.sum()} / 314 ({labeled_mask.mean():.1%}) have family-level taxonomy")
print(f"Unique phyla: {len(set(x for x in tax_arrays['phylum'] if x is not None))}")
print(f"Unique families: {len(set(x for x in tax_arrays['family'] if x is not None))}")
print(f"Unique genera: {len(set(x for x in tax_arrays['genus'] if x is not None))}")
```

- [ ] **Step 5: Add label-encoding cell for semi-supervised UMAP**

```python
# Encode family labels as integers. Merge rare families (< 3 MAGs) into "Other".
family_counts = pd.Series([f for f in tax_arrays["family"] if f is not None]).value_counts()
rare_families = set(family_counts[family_counts < 3].index)
print(f"Rare families merged into 'Other': {len(rare_families)}")

family_for_encode = []
for f in tax_arrays["family"]:
    if f is None:
        family_for_encode.append(None)
    elif f in rare_families:
        family_for_encode.append("Other")
    else:
        family_for_encode.append(f)

# Build integer codes: -1 for unlabeled (UMAP's semi-supervised convention)
le = LabelEncoder()
labeled_families = [f for f in family_for_encode if f is not None]
le.fit(labeled_families)

family_codes = np.full(314, -1, dtype=int)
for i, f in enumerate(family_for_encode):
    if f is not None:
        family_codes[i] = le.transform([f])[0]

assert (family_codes == -1).sum() == (~labeled_mask).sum()
print(f"Family codes: {len(le.classes_)} classes after rare-merging; -1 used for {(family_codes == -1).sum()} unlabeled MAGs")
```

- [ ] **Step 6: Add PCA baseline cell**

```python
pca_2 = PCA(n_components=2, random_state=42).fit_transform(X)
pca_10 = PCA(n_components=10, random_state=42).fit_transform(X)

assert pca_2.shape == (314, 2) and pca_10.shape == (314, 10)
np.save(EMB_DIR / "pca_2d.npy", pca_2)
np.save(EMB_DIR / "pca_10d.npy", pca_10)

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(pca_2[:, 0], pca_2[:, 1], c=metadata["gc_content"], cmap="viridis", s=20)
ax.set_title("PCA (colored by GC)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.colorbar(sc, label="GC content")
plt.tight_layout(); plt.show()
```

- [ ] **Step 7: Run all cells, verify**

Expected output:
- "Loaded X shape=(314, 256), mag_ids[0]=spire_mag_01923446"
- "Samples: 41 unique sample_ids across 314 MAGs"
- "Labeled MAGs: 126 / 314 (40.1%) have family-level taxonomy" (or similar)
- "Family codes: N classes..." prints
- PCA scatter rendered
- Files exist: `data/embeddings/pca_2d.npy`, `data/embeddings/pca_10d.npy`

```bash
ls -l data/embeddings/
```
Both `.npy` files should be present.

---

## Task 2: NMF (10-d only, for clustering)

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 2)

NMF n_components=10 on row-normalized k-mer frequencies. No 2-component fit — too coarse for visualization. The 10-d representation feeds k-means in Task 8.

- [ ] **Step 1: Add markdown section header**

```markdown
## 2. NMF — Non-negative Matrix Factorization (10-d, for clustering)

k-mer frequencies are non-negative by construction. NMF gives parts-based components ("k-mer signatures") that PCA cannot. Fit n=10 for downstream clustering (parallel to PCA-10d). No 2-component scatter — too coarse for visualization.
```

- [ ] **Step 2: Add NMF fitting cell**

```python
nmf_10 = NMF(n_components=10, init="nndsvd", max_iter=1000, random_state=42)
W_10 = nmf_10.fit_transform(X)  # (314, 10)

assert W_10.shape == (314, 10) and (W_10 >= 0).all(), "NMF output must be non-negative"
print(f"NMF-10 reconstruction error: {nmf_10.reconstruction_err_:.4f}")

np.save(EMB_DIR / "nmf_10d.npy", W_10)
```

- [ ] **Step 3: Add interpretability cell (top k-mers per component)**

```python
from itertools import product
kmer_names = ["".join(p) for p in product("ACGT", repeat=4)]
assert len(kmer_names) == 256

H = nmf_10.components_  # (10, 256)
print("Top 5 k-mers per NMF component:")
for k in range(10):
    top_idx = np.argsort(H[k])[::-1][:5]
    top_kmers = [kmer_names[i] for i in top_idx]
    print(f"  Component {k}: {top_kmers}")
```

- [ ] **Step 4: Run cells, verify**

Expected:
- No errors; finite reconstruction error
- 10 lines of top-k-mer printouts
- File exists: `data/embeddings/nmf_10d.npy`

If sklearn warns about non-convergence after `max_iter=1000`, bump to 2000 and re-run.

---

## Task 3: Autoencoder

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 3)

PyTorch autoencoder `256 → 64 → 16 → 2 → 16 → 64 → 256`, MSE loss, Adam, 200 epochs. The 2D bottleneck activations become the embedding.

- [ ] **Step 1: Add markdown section header**

```markdown
## 3. Autoencoder (2-d bottleneck)

Non-linear bottleneck embedding. PyTorch MLP with 2D bottleneck. Captures k-mer interactions PCA/NMF cannot model. Likely to underperform UMAP at N=314 — comparison is the point.
```

- [ ] **Step 2: Add model + training cell**

```python
class Autoencoder(nn.Module):
    def __init__(self, in_dim=256, bottleneck=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(),
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


device = "cpu"
torch.manual_seed(42)
model = Autoencoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_t = torch.from_numpy(X.astype(np.float32)).to(device)
n = X_t.shape[0]
batch_size = 32
n_epochs = 200

losses = []
for epoch in range(n_epochs):
    perm = torch.randperm(n)
    epoch_loss = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        batch = X_t[idx]
        opt.zero_grad()
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * batch.shape[0]
    losses.append(epoch_loss / n)
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:3d}  loss={losses[-1]:.6f}")
```

- [ ] **Step 3: Add embedding extraction + collapse check cell**

```python
model.eval()
with torch.no_grad():
    _, Z = model(X_t)
ae_2 = Z.cpu().numpy()

assert ae_2.shape == (314, 2)

std0, std1 = ae_2[:, 0].std(), ae_2[:, 1].std()
print(f"AE bottleneck std: dim0={std0:.4f}, dim1={std1:.4f}")
if std0 < 0.01 or std1 < 0.01:
    print("WARNING: bottleneck appears collapsed. Retrain with seed=123 before proceeding.")

np.save(EMB_DIR / "autoencoder_2d.npy", ae_2)
```

- [ ] **Step 4: Add loss curve + scatter cell**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(losses)
axes[0].set_title("AE training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")

sc = axes[1].scatter(ae_2[:, 0], ae_2[:, 1], c=metadata["gc_content"], cmap="viridis", s=20)
axes[1].set_title("Autoencoder bottleneck (colored by GC)")
axes[1].set_xlabel("z1"); axes[1].set_ylabel("z2")
plt.colorbar(sc, ax=axes[1], label="GC content")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Run cells, verify**

Expected:
- Loss decreases over epochs
- Both bottleneck dims std > 0.01 (else re-run with seed=123)
- Loss curve + scatter rendered
- File exists: `data/embeddings/autoencoder_2d.npy`

---

## Task 4: UMAP hyperparameter sweep

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 4)

12-point grid `n_neighbors × min_dist`, selected by k-means(k=4) Silhouette on each 2D embedding.

- [ ] **Step 1: Add markdown section header**

```markdown
## 4. UMAP hyperparameter sweep

Grid: `n_neighbors ∈ {5, 15, 30, 50}` × `min_dist ∈ {0.0, 0.1, 0.5}`. Selection: Silhouette of k-means(k=4).
```

- [ ] **Step 2: Add sweep cell**

```python
sweep_results = []
sweep_embeddings = {}

for n_nb in [5, 15, 30, 50]:
    for md in [0.0, 0.1, 0.5]:
        reducer = umap.UMAP(n_neighbors=n_nb, min_dist=md, n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(emb)
        sil = silhouette_score(emb, km.labels_)
        sweep_results.append({"n_neighbors": n_nb, "min_dist": md, "silhouette_kmeans4": sil})
        sweep_embeddings[(n_nb, md)] = emb
        print(f"n_neighbors={n_nb:3d}  min_dist={md:.1f}  silhouette={sil:.4f}")

sweep_df = pd.DataFrame(sweep_results).sort_values("silhouette_kmeans4", ascending=False)
sweep_df.to_csv(DATA_DIR / "umap_sweep_results.csv", index=False)
print("\nTop 3:\n", sweep_df.head(3))
```

- [ ] **Step 3: Add best-pick cell**

```python
best_row = sweep_df.iloc[0]
best_nb = int(best_row["n_neighbors"])
best_md = float(best_row["min_dist"])
umap_tuned_2 = sweep_embeddings[(best_nb, best_md)]

assert umap_tuned_2.shape == (314, 2)
np.save(EMB_DIR / "umap_tuned_2d.npy", umap_tuned_2)
print(f"Best UMAP params: n_neighbors={best_nb}, min_dist={best_md}, silhouette={best_row['silhouette_kmeans4']:.4f}")
```

- [ ] **Step 4: Add scatter cell**

```python
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(umap_tuned_2[:, 0], umap_tuned_2[:, 1], c=metadata["gc_content"], cmap="viridis", s=20)
ax.set_title(f"UMAP tuned (n_nb={best_nb}, min_dist={best_md}, colored by GC)")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
plt.colorbar(sc, label="GC content")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Run cells, verify**

Expected:
- 12 grid lines print
- `data/umap_sweep_results.csv` saved
- File exists: `data/embeddings/umap_tuned_2d.npy`

---

## Task 5: Semi-supervised UMAP

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 5)

UMAP with `y=family_codes` (`-1` for unlabeled), using best hyperparameters from Task 4 so the comparison isolates the effect of labels.

- [ ] **Step 1: Add markdown section header**

```markdown
## 5. Semi-supervised UMAP

UMAP with family labels (-1 for unlabeled 188 MAGs). Uses best n_neighbors/min_dist from §4 to isolate the label effect. Caveat: evaluating taxonomic Silhouette on the same labels used for fitting is circular — flagged in writeup.
```

- [ ] **Step 2: Add semi-supervised UMAP cell**

```python
reducer_semi = umap.UMAP(
    n_neighbors=best_nb,
    min_dist=best_md,
    n_components=2,
    random_state=42,
)
umap_semisup_2 = reducer_semi.fit_transform(X, y=family_codes)

assert umap_semisup_2.shape == (314, 2)
np.save(EMB_DIR / "umap_semisup_2d.npy", umap_semisup_2)
```

- [ ] **Step 3: Add scatter cell colored by family (labeled subset) with unlabeled in gray**

```python
fig, ax = plt.subplots(figsize=(7, 5))

# Plot unlabeled first in gray
unlab = ~labeled_mask
ax.scatter(umap_semisup_2[unlab, 0], umap_semisup_2[unlab, 1],
           c="lightgray", s=15, label=f"unlabeled (n={unlab.sum()})")

# Plot labeled colored by family code
lab_idx = np.where(labeled_mask)[0]
sc = ax.scatter(umap_semisup_2[lab_idx, 0], umap_semisup_2[lab_idx, 1],
                c=family_codes[lab_idx], cmap="tab20", s=25)
ax.set_title("Semi-supervised UMAP (labeled colored by family, unlabeled gray)")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(loc="best", fontsize=8)
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Run cells, verify**

Expected:
- No UMAP error
- Scatter rendered with both gray and colored points
- File exists: `data/embeddings/umap_semisup_2d.npy`

If UMAP errors with "Found array with NaN" or similar, check that `family_codes` is `int` and has no NaN — the encoding in Task 1 step 5 already enforces this. If a specific family has only 1 MAG after rare-merging, increase the rare threshold from 3 to 5 in Task 1 step 5 and re-run.

---

## Task 6: Agglomerative clustering + dendrogram

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 6)

Hierarchical clustering on cosine distance over 4-mer frequencies, average linkage. Dendrogram with red cut line saved as PNG.

- [ ] **Step 1: Add markdown section header**

```markdown
## 6. Agglomerative clustering (cosine + average linkage)

Lecture 7 "q-gram cosine" agglomerative clustering on 4-mer frequencies.
```

- [ ] **Step 2: Add linkage cell**

```python
D = cosine_distances(X)  # (314, 314)
condensed = sch.distance.squareform(D, checks=False)
Z_link = sch.linkage(condensed, method="average")
```

- [ ] **Step 3: Add best-k selection cell**

```python
best_k_agg, best_sil_agg, best_labels_agg = None, -1.0, None
for k in range(2, 11):
    labels = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(D)
    sil = silhouette_score(D, labels, metric="precomputed")
    print(f"  k={k}  silhouette={sil:.4f}")
    if sil > best_sil_agg:
        best_sil_agg, best_k_agg, best_labels_agg = sil, k, labels

print(f"\nBest agglomerative: k={best_k_agg}, silhouette={best_sil_agg:.4f}")
np.save(CLU_DIR / "agglomerative_labels.npy", best_labels_agg)
```

- [ ] **Step 4: Add dendrogram-with-cut cell**

```python
heights = sorted(Z_link[:, 2])
cut_height = heights[-(best_k_agg - 1)] if best_k_agg > 1 else heights[-1]

fig, ax = plt.subplots(figsize=(12, 5))
sch.dendrogram(Z_link, no_labels=True, ax=ax, color_threshold=cut_height)
ax.axhline(cut_height, color="red", linestyle="--", linewidth=1)
ax.set_title(f"Dendrogram with flat cut at k={best_k_agg} (height={cut_height:.4f})")
ax.set_ylabel("cosine distance")
plt.tight_layout()
plt.savefig(DATA_DIR / "dendrogram.png", dpi=150)
plt.show()
```

- [ ] **Step 5: Run cells, verify**

Expected:
- 9 lines of k-Silhouette printouts
- Best k reported (likely 2–5)
- Files exist: `data/clusters/agglomerative_labels.npy`, `data/dendrogram.png`

---

## Task 7: HDBSCAN on tuned UMAP

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 7)

- [ ] **Step 1: Add markdown section header**

```markdown
## 7. HDBSCAN on tuned UMAP

Density-based clustering with noise label (-1). Run on the best UMAP-2D from §4.
```

- [ ] **Step 2: Add HDBSCAN cell with all-noise fallback**

```python
def run_hdbscan(embedding, min_cluster_size, min_samples):
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    ).fit_predict(embedding)

hdb_labels = run_hdbscan(umap_tuned_2, min_cluster_size=10, min_samples=5)
n_noise = int((hdb_labels == -1).sum())
n_clusters = len(set(hdb_labels) - {-1})
print(f"min_cluster_size=10 → {n_clusters} clusters, {n_noise} noise points")

if n_noise == 314:
    print("All points labeled noise. Retrying with min_cluster_size=5.")
    hdb_labels = run_hdbscan(umap_tuned_2, min_cluster_size=5, min_samples=3)
    n_noise = int((hdb_labels == -1).sum())
    n_clusters = len(set(hdb_labels) - {-1})
    print(f"min_cluster_size=5 → {n_clusters} clusters, {n_noise} noise points")

np.save(CLU_DIR / "hdbscan_labels.npy", hdb_labels)
```

- [ ] **Step 3: Add Silhouette-excluding-noise cell**

```python
mask = hdb_labels != -1
if mask.sum() > 1 and n_clusters >= 2:
    hdb_silhouette = silhouette_score(umap_tuned_2[mask], hdb_labels[mask])
    print(f"HDBSCAN silhouette (noise excluded): {hdb_silhouette:.4f}")
    print(f"Noise fraction: {n_noise}/{len(hdb_labels)} = {n_noise/len(hdb_labels):.2%}")
else:
    hdb_silhouette = float("nan")
    print("Cannot compute silhouette: fewer than 2 non-noise clusters.")
```

- [ ] **Step 4: Add scatter cell (noise gray)**

```python
fig, ax = plt.subplots(figsize=(6, 5))
for lab in sorted(set(hdb_labels)):
    pts = umap_tuned_2[hdb_labels == lab]
    if lab == -1:
        ax.scatter(pts[:, 0], pts[:, 1], c="lightgray", s=20, label=f"noise ({len(pts)})")
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=20, label=f"cluster {lab} ({len(pts)})")
ax.legend(fontsize=8, loc="best")
ax.set_title("UMAP + HDBSCAN clusters")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Run cells, verify**

Expected:
- N clusters, M noise printed
- Silhouette score in [-1, 1] or NaN
- Scatter rendered
- File exists: `data/clusters/hdbscan_labels.npy`

---

## Task 8: Comparison — internal Silhouette, taxonomic Silhouette, cluster purity, sample confound, scatter grids, ARI

**Files:**
- Modify: `03_advanced_methods.ipynb` (append section 8)

The unified evaluation. Pulls all saved embeddings and cluster labels and produces five comparison artifacts plus the final discussion writeup.

- [ ] **Step 1: Add markdown section header**

```markdown
## 8. Comparison: Silhouette (internal + taxonomic), purity, sample confound, scatter grids, ARI
```

- [ ] **Step 2: Add per-method k-means clustering + internal + taxonomic Silhouette cell**

```python
embeddings_for_eval = {
    "PCA-10d": pca_10,
    "NMF-10d": W_10,
    "Autoencoder-2d": ae_2,
    "UMAP-tuned-2d": umap_tuned_2,
    "UMAP-semisup-2d": umap_semisup_2,
}

records = []
best_kmeans_labels = {}  # for ARI, purity, grid

for name, emb in embeddings_for_eval.items():
    # Pick best k by internal Silhouette
    best_k, best_sil, best_labels = None, -1.0, None
    for k in [2, 3, 4, 5, 6, 8, 10]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
        sil = silhouette_score(emb, km.labels_)
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, km.labels_
    best_kmeans_labels[name] = best_labels
    np.save(CLU_DIR / f"kmeans_labels_{name.replace('-', '_')}.npy", best_labels)

    # Taxonomic Silhouette on labeled subset
    row = {"method": name, "clusterer": "kmeans", "k": best_k,
           "silhouette_internal": best_sil, "n_noise": 0}
    for level in ["phylum", "family", "genus"]:
        level_mask = np.array([tax_arrays[level][i] is not None for i in range(314)])
        level_subset_labels = best_labels[level_mask]
        level_subset_taxa = np.array([tax_arrays[level][i] for i in np.where(level_mask)[0]])
        # Need at least 2 distinct taxa in the subset for a meaningful score
        n_distinct = len(set(level_subset_taxa))
        if n_distinct >= 2 and len(level_subset_labels) > n_distinct:
            row[f"silhouette_{level}"] = silhouette_score(
                emb[level_mask], pd.Series(level_subset_taxa).astype("category").cat.codes
            )
        else:
            row[f"silhouette_{level}"] = float("nan")
    records.append(row)

# Agglomerative — same eval
records.append({
    "method": "4mer-cosine", "clusterer": "agglomerative", "k": best_k_agg,
    "silhouette_internal": best_sil_agg, "n_noise": 0,
    "silhouette_phylum": float("nan"),  # filled below
    "silhouette_family": float("nan"),
    "silhouette_genus": float("nan"),
})
for level in ["phylum", "family", "genus"]:
    level_mask = np.array([tax_arrays[level][i] is not None for i in range(314)])
    if level_mask.sum() > 2:
        taxa = np.array([tax_arrays[level][i] for i in np.where(level_mask)[0]])
        codes = pd.Series(taxa).astype("category").cat.codes
        if len(set(taxa)) >= 2:
            # Use the agglomerative labels restricted to the labeled subset, evaluated in 4-mer space
            records[-1][f"silhouette_{level}"] = silhouette_score(
                X[level_mask], codes
            )

# HDBSCAN — same eval (no taxonomic Silhouette for HDBSCAN since noise complicates labeled-subset alignment)
records.append({
    "method": "UMAP-tuned-2d", "clusterer": "hdbscan", "k": n_clusters,
    "silhouette_internal": hdb_silhouette, "n_noise": n_noise,
    "silhouette_phylum": float("nan"),
    "silhouette_family": float("nan"),
    "silhouette_genus": float("nan"),
})

results_df = pd.DataFrame(records)
results_df.to_csv(DATA_DIR / "silhouette_results.csv", index=False)
print(results_df.to_string(index=False))
```

- [ ] **Step 3: Add cluster taxonomic purity + AMI cell (labeled subset)**

```python
purity_records = []

for name, labels in best_kmeans_labels.items():
    # Restrict to labeled MAGs (family present)
    fam_present = np.array([tax_arrays["family"][i] is not None for i in range(314)])
    if fam_present.sum() < 2:
        continue
    families_sub = np.array([tax_arrays["family"][i] for i in np.where(fam_present)[0]])
    labels_sub = labels[fam_present]

    ami = adjusted_mutual_info_score(families_sub, families_sub)  # placeholder, replaced below
    ami = adjusted_mutual_info_score(
        pd.Series(families_sub).astype("category").cat.codes,
        labels_sub,
    )

    for cid in sorted(set(labels_sub)):
        mask_c = labels_sub == cid
        n_in_cluster = int(mask_c.sum())
        if n_in_cluster == 0:
            continue
        fam_counts = pd.Series(families_sub[mask_c]).value_counts()
        dominant = fam_counts.index[0]
        purity = float(fam_counts.iloc[0] / n_in_cluster)
        purity_records.append({
            "method": name,
            "cluster_id": int(cid),
            "n_mags_labeled": n_in_cluster,
            "dominant_family": dominant,
            "purity": purity,
            "ami_overall": ami,
        })

purity_df = pd.DataFrame(purity_records)
purity_df.to_csv(DATA_DIR / "cluster_purity.csv", index=False)
print("Cluster purity (head):")
print(purity_df.head(20).to_string(index=False))

print("\nMethod-level summary (mean purity, AMI):")
print(purity_df.groupby("method").agg(
    mean_purity=("purity", "mean"),
    ami_overall=("ami_overall", "first"),
).round(3).to_string())
```

- [ ] **Step 4: Add sample/host confound ARI cell (full dataset)**

```python
sample_codes = pd.Series(sample_ids).astype("category").cat.codes.values

confound_records = []
for name, labels in best_kmeans_labels.items():
    sample_ari = adjusted_rand_score(sample_codes, labels)
    confound_records.append({
        "method": name, "n_samples": int(len(set(sample_ids))), "sample_ari": sample_ari
    })

# Also for agglomerative + hdbscan
confound_records.append({
    "method": "Agglomerative", "n_samples": int(len(set(sample_ids))),
    "sample_ari": adjusted_rand_score(sample_codes, best_labels_agg),
})
confound_records.append({
    "method": "UMAP+HDBSCAN", "n_samples": int(len(set(sample_ids))),
    "sample_ari": adjusted_rand_score(sample_codes, hdb_labels),
})

confound_df = pd.DataFrame(confound_records)
confound_df.to_csv(DATA_DIR / "sample_confound.csv", index=False)
print("Sample confound ARI (higher = more host/batch confound):")
print(confound_df.sort_values("sample_ari", ascending=False).to_string(index=False))
```

- [ ] **Step 5: Add Silhouette + sample-ARI bar chart cell**

```python
fig, ax = plt.subplots(figsize=(12, 6))

# Build the bar groups: per method, three bars (internal Silhouette, family-tax Silhouette, sample ARI).
methods_in_order = [r["method"] + f" ({r['clusterer']})" for r in records]
internal_scores = [r["silhouette_internal"] for r in records]
family_scores = [r.get("silhouette_family", float("nan")) for r in records]

# Sample ARI: align by method name
sample_ari_lookup = dict(zip(confound_df["method"], confound_df["sample_ari"]))
sample_aris = []
for r in records:
    # Map record method label to confound_df key
    if r["clusterer"] == "agglomerative":
        sample_aris.append(sample_ari_lookup.get("Agglomerative", float("nan")))
    elif r["clusterer"] == "hdbscan":
        sample_aris.append(sample_ari_lookup.get("UMAP+HDBSCAN", float("nan")))
    else:
        sample_aris.append(sample_ari_lookup.get(r["method"], float("nan")))

x = np.arange(len(methods_in_order))
w = 0.25
ax.bar(x - w, internal_scores, w, label="Silhouette (internal)", color="#4C72B0")
ax.bar(x, family_scores, w, label="Silhouette (family, labeled subset)", color="#55A868")
ax.bar(x + w, sample_aris, w, label="Sample ARI (confound; lower is better)", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(methods_in_order, rotation=30, ha="right", fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Score")
ax.set_title("Method comparison: internal vs taxonomic Silhouette and sample confound")
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "silhouette_advanced_comparison.png", dpi=150)
plt.show()
```

- [ ] **Step 6: Add three scatter grid variants cell**

```python
panels = [
    ("PCA", pca_2, best_kmeans_labels["PCA-10d"]),
    ("NMF (PCA-2d layout, NMF-10d clusters)", pca_2, best_kmeans_labels["NMF-10d"]),
    ("Autoencoder", ae_2, best_kmeans_labels["Autoencoder-2d"]),
    ("UMAP tuned", umap_tuned_2, best_kmeans_labels["UMAP-tuned-2d"]),
    ("UMAP semi-sup", umap_semisup_2, best_kmeans_labels["UMAP-semisup-2d"]),
    ("UMAP + HDBSCAN", umap_tuned_2, hdb_labels),
]


def draw_grid(panels, color_source, color_kind, savepath):
    """color_kind in {'cluster', 'phylum', 'gc'}"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, emb, labels) in zip(axes.flat, panels):
        if color_kind == "cluster":
            if -1 in set(labels):
                m = labels == -1
                ax.scatter(emb[m, 0], emb[m, 1], c="lightgray", s=15)
                ax.scatter(emb[~m, 0], emb[~m, 1], c=labels[~m], cmap="tab10", s=15)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=15)
        elif color_kind == "phylum":
            unlab = np.array([tax_arrays["phylum"][i] is None for i in range(314)])
            ax.scatter(emb[unlab, 0], emb[unlab, 1], c="lightgray", s=12)
            lab = ~unlab
            phyla_for_color = np.array([tax_arrays["phylum"][i] for i in np.where(lab)[0]])
            codes = pd.Series(phyla_for_color).astype("category").cat.codes
            ax.scatter(emb[lab, 0], emb[lab, 1], c=codes, cmap="tab20", s=18)
        elif color_kind == "gc":
            ax.scatter(emb[:, 0], emb[:, 1], c=metadata["gc_content"], cmap="viridis", s=15)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Embeddings colored by {color_kind}", fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.show()


draw_grid(panels, None, "cluster", DATA_DIR / "embeddings_advanced_grid_kmeans.png")
draw_grid(panels, None, "phylum",  DATA_DIR / "embeddings_advanced_grid_phylum.png")
draw_grid(panels, None, "gc",      DATA_DIR / "embeddings_advanced_grid_gc.png")
```

- [ ] **Step 7: Add ARI heatmap cell**

```python
methods_for_ari = list(best_kmeans_labels.keys()) + ["UMAP+HDBSCAN", "Agglomerative"]
ari_labels = {**best_kmeans_labels,
              "UMAP+HDBSCAN": hdb_labels,
              "Agglomerative": best_labels_agg}

n_methods = len(methods_for_ari)
ari_mat = np.zeros((n_methods, n_methods))
for i, m1 in enumerate(methods_for_ari):
    for j, m2 in enumerate(methods_for_ari):
        ari_mat[i, j] = adjusted_rand_score(ari_labels[m1], ari_labels[m2])

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(ari_mat, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n_methods)); ax.set_yticks(range(n_methods))
ax.set_xticklabels(methods_for_ari, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(methods_for_ari, fontsize=9)
for i in range(n_methods):
    for j in range(n_methods):
        ax.text(j, i, f"{ari_mat[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(ari_mat[i, j]) > 0.5 else "black", fontsize=8)
plt.colorbar(im, label="ARI")
ax.set_title("Cross-method agreement (ARI)")
plt.tight_layout()
plt.savefig(DATA_DIR / "ari_matrix.png", dpi=150)
plt.show()
```

- [ ] **Step 8: Add final discussion markdown cell**

Markdown cell (user fills in `<...>` placeholders after running):

```markdown
## Discussion

**Best internal Silhouette:** `<METHOD>` at k=`<K>` with `<SCORE>`. Silhouette favors convex spherical clusters and reflects geometric separation in *that* method's projection, not biological truth.

**Best family-level taxonomic Silhouette** (on 126 labeled MAGs): `<METHOD>` with `<SCORE>`. (Semi-supervised UMAP's taxonomic Silhouette is excluded from fair comparison — labels were used during fitting.)

**Sample/host confound:** highest sample ARI = `<VALUE>` for `<METHOD>`. `<INTERPRETATION: ARI < 0.2 → low confound, methods clustering by biology; ARI > 0.5 → strong host/batch confound, take cluster interpretations with a grain of salt>`.

**Cluster purity:** mean family-purity per method:
- `<METHOD_1>`: `<purity>`
- `<METHOD_2>`: `<purity>`
- ...

**Cross-method ARI:** highest pair = `<PAIR>` (`<VAL>`); lowest = `<PAIR>` (`<VAL>`). High mutual ARI between unsupervised methods suggests the signal is robust across modeling families.

**HDBSCAN noise:** `<N_NOISE>/314` MAGs labeled noise. These tend to have `<OBSERVATION: small contigs / unusual GC / specific phylum>`.

**NMF interpretability:** top-loading 4-mers per component suggest `<COMPONENT 0 PATTERN, COMPONENT 1 PATTERN, ...>`. NMF provides this interpretability natively; PCA components have mixed-sign loadings.

**Limitations:**
- 60% of MAGs (188/314) lack taxonomy → taxonomic metrics are computed on a 126-MAG subset.
- N=314 is small; the Autoencoder bottleneck has limited capacity vs UMAP at this scale (confirmed by `<COMPARISON>`).
- Sample ARI is a confound *signal*, not a causal test — high ARI could also mean MAGs from the same host genuinely share biology.
```

- [ ] **Step 9: Run all section 8 cells, verify**

Expected:
- `silhouette_results.csv` printed: 7 rows (5 k-means + 1 agglomerative + 1 hdbscan), columns include `silhouette_phylum`, `silhouette_family`, `silhouette_genus`
- `cluster_purity.csv` saved; purity values in [0, 1]; method-level summary printed
- `sample_confound.csv` saved; ARI values in approximately [-0.1, 1]
- Three scatter grids saved (k-means, phylum, GC)
- ARI heatmap saved
- Discussion markdown rendered with `<...>` placeholders

```bash
ls /Users/danila/PycharmProjects/Big_Data/data/silhouette_results.csv \
   /Users/danila/PycharmProjects/Big_Data/data/cluster_purity.csv \
   /Users/danila/PycharmProjects/Big_Data/data/sample_confound.csv \
   /Users/danila/PycharmProjects/Big_Data/data/silhouette_advanced_comparison.png \
   /Users/danila/PycharmProjects/Big_Data/data/embeddings_advanced_grid_kmeans.png \
   /Users/danila/PycharmProjects/Big_Data/data/embeddings_advanced_grid_phylum.png \
   /Users/danila/PycharmProjects/Big_Data/data/embeddings_advanced_grid_gc.png \
   /Users/danila/PycharmProjects/Big_Data/data/ari_matrix.png \
   /Users/danila/PycharmProjects/Big_Data/data/dendrogram.png \
   /Users/danila/PycharmProjects/Big_Data/data/umap_sweep_results.csv
```

All ten paths should resolve.

- [ ] **Step 10: Fill in discussion placeholders**

Read the printed Silhouette table, sample confound table, purity summary, and ARI heatmap, then replace all `<...>` placeholders in the discussion cell with actual values from the run. Save the notebook.

---

## Self-review notes

**Spec coverage (each spec requirement → task):**
- NMF-10d (§NMF) → Task 2
- Autoencoder (§Autoencoder) → Task 3
- UMAP sweep (§UMAP sweep) → Task 4
- Semi-supervised UMAP (§Semi-supervised UMAP) → Task 5
- Agglomerative (§Agglomerative) → Task 6
- HDBSCAN (§HDBSCAN) → Task 7
- Internal Silhouette protocol → Task 8 step 2
- Taxonomic Silhouette at phylum/family/genus → Task 8 step 2
- Cluster purity + AMI → Task 8 step 3
- Sample/host confound ARI → Task 8 step 4
- `silhouette_results.csv` (long format with new tax columns) → Task 8 step 2
- `cluster_purity.csv` → Task 8 step 3
- `sample_confound.csv` → Task 8 step 4
- `silhouette_advanced_comparison.png` (with sample ARI bar) → Task 8 step 5
- `embeddings_advanced_grid_{kmeans,phylum,gc}.png` → Task 8 step 6
- `dendrogram.png` → Task 6 step 4
- `ari_matrix.png` → Task 8 step 7
- Failure handling: AE collapse → Task 3 step 3; HDBSCAN all-noise → Task 7 step 2; rare-family merge → Task 1 step 5; NMF non-convergence → Task 2 step 2 (max_iter=1000)
- PCA-2d/10d origin → Task 1 step 6
- Discussion writeup → Task 8 step 8 + step 10
- Stretch goals (QC filter, day-of-sampling, Parametric UMAP) → intentionally not in plan

**Placeholder scan:** Steps with `<METHOD>`, `<K>`, etc. appear only in the discussion markdown template (Task 8 step 8), which is explicit user-fill-in content after the run. Step 10 of Task 8 instructs how to replace them. These are intentional, not failures.

**Type/name consistency check:**
- `X` (L1-normalized, defined Task 1 step 3) — used in Tasks 2, 3, 4, 5, 6, 8
- `pca_2`, `pca_10` (Task 1 step 6) — used in Task 8 step 6 (grid layout)
- `W_10` (Task 2) — used in Task 8 step 2
- `ae_2` (Task 3) — used in Task 8 steps 2, 6
- `umap_tuned_2`, `best_nb`, `best_md` (Task 4) — used in Tasks 5, 7, 8
- `umap_semisup_2` (Task 5) — used in Task 8 steps 2, 6
- `best_labels_agg`, `best_k_agg`, `best_sil_agg` (Task 6) — used in Task 8 steps 2, 4, 7
- `hdb_labels`, `n_clusters`, `n_noise`, `hdb_silhouette` (Task 7) — used in Task 8 steps 2, 4, 6, 7
- `sample_ids`, `tax_arrays`, `labeled_mask`, `family_codes` (Task 1) — used in Tasks 5, 8
- `metadata` — used in scatter cells; consistent
- `EMB_DIR`, `CLU_DIR`, `DATA_DIR` — Path objects from Task 1 step 2, used everywhere
- `best_kmeans_labels` (built in Task 8 step 2) — used in Task 8 steps 3, 4, 6, 7
