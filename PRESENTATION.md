# Presentation Workflow — Metagenome Clustering via 4-mer Composition

---

## Overview

This project asks one central question:

> **Can we automatically group bacterial genomes by their biological identity using only the raw DNA sequence — no labels, no prior knowledge?**

The dataset is 314 **MAGs** (Metagenome-Assembled Genomes) — partial bacterial genomes reconstructed from a gut microbiome study (David et al. 2015, Bangladesh cholera cohort). Each MAG is a compressed representation of one bacterial strain living in a patient's gut, sampled across three disease phases: acute infection, early recovery, and late recovery.

---

## Section 1 — What is a MAG and why do we need to cluster them?

**Motivation:** When sequencing a gut microbiome sample, you don't get clean individual bacterial genomes — you get millions of fragmented DNA reads from hundreds of species mixed together. Computational tools reassemble these fragments into partial genomes called MAGs. The problem: MAGs have no labels. You don't know which bacterium each MAG belongs to, or whether two MAGs from different patients come from the same species. Clustering is the way to answer this without expensive wet-lab experiments.

**The dataset:** 314 MAGs from the David 2015 Bangladesh study, stored as compressed FASTA files (`.fa.gz`). Each file contains DNA contigs — partial chromosome fragments — that were assembled from sequencing reads.

> **Supporting research:** Metagenome-assembled genomes have become the standard unit of metagenomic analysis since 2013 (Parks et al. 2015, CheckM). The SPIRE database (Fullam et al. 2023) hosts quality-controlled MAG collections from hundreds of microbiome studies globally.

**Transition:** To cluster MAGs, we need to turn raw DNA sequences into numbers a computer can compare. That's the feature engineering step.

---

## Section 2 — Feature Engineering: 4-mer Frequencies (Notebook 01)

**Motivation:** You can't directly compare DNA sequences of different lengths. You need a fixed-size numeric representation that captures something meaningful about each genome's composition. The approach used here — **k-mer frequency profiling** — is one of the most established methods in computational genomics.

**What is a k-mer?** A k-mer is every possible substring of length k from a DNA sequence. For k=4, the vocabulary is all combinations of A, C, G, T taken 4 at a time: AAAA, AAAC, AAAG, ... = **256 unique 4-mers**.

**What we do:** For each MAG, slide a window of size 4 across all contigs and count how often each of the 256 4-mers appears. Then divide by the total count to get frequencies that sum to 1. The result is one 256-dimensional vector per MAG.

```
MAG genome → count all 4-mers → normalize → 256-dim frequency vector
314 MAGs   →                                 (314 × 256) feature matrix
```

**Why does this work?** Different bacterial species have distinct DNA composition preferences — what's called "genomic signature." A bacterium from the Bacteroidota phylum naturally uses different k-mer patterns than one from Proteobacteria, even in regions of the genome that don't code for proteins. This signature is stable within a species and variable between species.

> **Supporting research:** Genomic signatures via k-mer composition were first described by Karlin & Burge (1995) and shown to be genus/species-specific. Applied to metagenomics binning by Teeling et al. (2004). 4-mers specifically are the standard choice — long enough to be specific, short enough to count reliably on partial genomes.

**Key output:** `kmer4_features.npy` — shape (314, 256). Each row sums to 1.0.

**Also computed as a by-product:** genome-level metadata extracted directly from parsing the FASTA files:
- GC content (fraction of G and C bases in the genome)
- Genome size (total base pairs)
- Number of contigs
- Mean and max contig length

Saved to `metadata.csv` — used throughout all downstream analysis for coloring and validation.

**Conclusion:** We now have a (314, 256) feature matrix where each row is a numerical fingerprint of one MAG. The matrix is ready for dimensionality reduction and clustering.

---

## Section 3 — Metadata: Where the Labels Come From

**Motivation:** Clustering is unsupervised — the algorithm doesn't see labels. But to evaluate whether our clusters are biologically meaningful, we need some ground truth to compare against. We have two sources:

**Source 1 — Disease phase labels** (`mag_metadata_full.csv`): Each MAG was sequenced from a patient stool sample collected at a specific timepoint. By tracing MAG identifiers through the SPIRE database → ENA accession PRJEB9150 → David 2015 Table S2, we recovered the collection day and disease phase for all 314 MAGs:
- **Acute** (day 0–1): 116 MAGs — peak cholera infection
- **Early recovery** (day 5–9): 82 MAGs
- **Late recovery** (day 26–31): 116 MAGs

**Source 2 — Taxonomy** (`spire_taxonomy.csv`): SPIRE runs GTDB-Tk (a standard taxonomy classifier) on each MAG. However, GTDB-Tk only classifies genomes with ≥50% completeness. Only **126 of 314 MAGs** pass this threshold — the other 188 are too fragmented. Among the 126 classified:
- Bacteroidota: 48 MAGs
- Firmicutes_A: 27 MAGs
- Proteobacteria: 20 MAGs
- + 5 smaller phyla

These labels are **never shown to the clustering algorithms** — they are only used afterwards to check whether our clusters match reality.

---

## Section 4 — Hypotheses

Before running any analysis, four hypotheses were formulated:

| ID | Hypothesis |
|---|---|
| **H1** | MAGs from the acute phase are separable from recovery-phase MAGs in k-mer space (disease succession leaves a genomic signature) |
| **H2** | MAGs from the same bacterial group (phylum) cluster together in k-mer space |
| **H3** | GC content — the fraction of G and C bases — co-varies with k-mer clusters |
| **H4** | 4-mers capture more structure than shorter k-mers (k=2 or k=3) |

> **Why H1?** Cholera kills anaerobic gut bacteria and creates an oxygen-rich environment where aerobic bacteria (Proteobacteria) bloom. As the patient recovers, anaerobes return. If this succession leaves a k-mer signature, we should be able to detect it without labels. (Hsiao et al. 2014; Subramanian et al. 2014 — gut microbiome recovery after diarrheal disease.)

> **Why H2?** Genomic signature is known to be phylum-level specific (Karlin 1999). If 4-mers carry taxonomic signal, same-phylum MAGs should naturally cluster.

> **Why H3?** GC content is the single strongest determinant of k-mer composition. Bacteroidota average ~46% GC, Actinobacteriota ~60%, Firmicutes ~35%.

> **Why H4?** Longer k-mers encode more specific sequence context. But longer k-mers also become sparser on short/fragmented genomes.

---

## Section 5 — Basic Embeddings: PCA, t-SNE, UMAP (Notebook 02)

**Motivation:** A 256-dimensional vector can't be visualized or clustered easily. We reduce it to 2D using three different methods, each with a different philosophy — then compare which reveals the most structure.

---

### 5.1 — PCA (Principal Component Analysis)

**What it does:** Finds the directions of maximum variance in the data. Linear — every point is a weighted sum of the original features, rearranged so the first axis captures as much variation as possible.

**Key findings:**
- **PC1 alone explains 43.8% of all variance** — the data has one dominant axis
- Only 7 components needed for 80% of variance, 11 for 90% — the 256-dim space is intrinsically very low-dimensional
- PC1 is a **pure GC content axis**: AT-rich k-mers (ATAA, TTAT) load positively; GC-rich k-mers (CACC, ACCG) load negatively
- Scatter plot shows clean left-to-right gradient by GC bin

**What this means:** The single strongest signal in bacterial k-mer composition is GC content. Everything else is secondary.

---

### 5.2 — t-SNE

**What it does:** A non-linear method that preserves *local* structure — points that are similar in 256D land near each other in 2D, but global distances are distorted. Run on the top 50 PCA components (standard practice to reduce noise before t-SNE).

**Key findings:**
- Reveals sub-clusters hidden inside the PCA blobs — GC groups that looked uniform in PCA split into multiple tight groups in t-SNE
- Silhouette (k-means) = **0.51** — better than raw PCA (0.40)
- Shows that within each GC level there is additional fine-grained structure

---

### 5.3 — UMAP

**What it does:** Also non-linear, but unlike t-SNE it better preserves *global* structure (overall distances between clusters). Faster and more consistent than t-SNE.

**Key findings:**
- Strongest cluster structure — Silhouette (k-means) = **0.65**
- Reveals a clear **global two-region split** with a visible gap between the upper and lower halves
- Two completely isolated points far from everything else (later identified as *Porphyromonas* genus — more below)
- Despite clean clustering, **GC silhouette = 0.06** — GC content does not explain what UMAP finds

**The key tension:** UMAP finds excellent clusters (0.65) but GC content doesn't explain them (0.06). Something more specific is driving the clustering. This motivates the taxonomy analysis.

---

### 5.4 — H1 Test: Disease Phase

Colored all three embeddings by disease phase (acute / early recovery / late recovery). Silhouette scores vs phase were near zero across all embeddings — **acute and recovery MAGs interleave, they do not separate**.

**H1 result: Not supported.** Disease phase is not encoded in 4-mer composition at the genome level. This makes biological sense: a MAG's k-mer signature is fixed by its evolutionary history (which species it is), not by which day it was sampled.

---

### 5.5 — H2 Test: Taxonomy (Phylum)

Colored the 126 labeled MAGs by phylum. Clear findings:
- **Extreme-GC phyla separate cleanly**: Actinobacteriota (high GC) and Campylobacterota (low GC) form distinct isolated clusters in all three embeddings
- **Moderate-GC phyla partially mix**: Bacteroidota, Firmicutes_A, and Proteobacteria (all ~45–55% GC) overlap — but sub-structure within them is visible in UMAP
- The **strongest single evidence for H2**: two isolated UMAP points identified as 12 MAGs all belonging to the *Porphyromonas* genus — UMAP isolated them without ever seeing a taxonomy label

**H2 result: Partially supported.** Taxonomy is reflected in k-mer space — strongly for phyla with extreme GC, partially for moderate-GC phyla.

---

### 5.6 — H3 Test: GC Content

PC1 is literally GC content. PCA silhouette vs GC bins = **0.20**. The global two-region UMAP split corresponds to high-GC (mean 0.499) vs low-GC (mean 0.413) genomes, with phyla sorting accordingly.

**H3 result: Strongly supported.**

---

### 5.7 — H4 Test: k-mer Length

Raw silhouette scores on scaled feature matrices:
- k=2 (16 features): 0.275
- k=3 (64 features): 0.240
- k=4 (256 features): 0.237

**H4 result: Rejected.** More k-mer dimensions do not improve raw clustering. GC content — which 2-mers already capture — is the dominant signal. 4-mers are still preferred for downstream methods because they capture *more patterns*, but they don't automatically produce better clusters on their own.

**Transition:** Basic embeddings reveal that k-mer composition carries biological signal, but GC content dominates everything. The next step is to apply more sophisticated methods that can look past GC content and find finer taxonomic structure.

---

## Section 6 — Advanced Embeddings and Clustering (Notebook 03)

**Motivation:** PCA and basic UMAP are good starting points, but they have limitations. PCA is linear and dominated by GC content. UMAP with default settings finds clean clusters but we don't know if they align with real taxonomy. The goal of this notebook is to:
1. Try methods that can look past the GC signal (NMF)
2. Tune UMAP properly instead of using defaults
3. Use density-based clustering that doesn't assume spherical clusters
4. Rigorously evaluate every method against real taxonomy labels

All methods use **L1-normalized** k-mer frequencies (rows summing to 1) as input.

---

### 6.1 — NMF: Non-negative Matrix Factorization (10 dimensions)

**What it does:** Decomposes each MAG's k-mer profile into a weighted sum of 10 basis patterns — all weights non-negative. Unlike PCA where components can cancel each other out, NMF represents each genome as a *mixture* of k-mer signatures. Think of it as: "this genome is 70% AT-rich pattern + 30% GC-rich pattern."

**Why better than PCA for this data:** k-mer frequencies are non-negative by definition. NMF respects this constraint; PCA does not. NMF components are interpretable as real k-mer usage styles.

**What the components found:** 10 distinct k-mer patterns emerged:
- Components 2, 3, 5 → AT-rich (low GC): poly-A/T repeats, alternating AT
- Components 6, 8 → high GC: alternating GC dinucleotides (Actinobacteria signature), stacked CG repeats
- Most top k-mer pairs are reverse complements of each other (AAAA/TTTT, GCGC/CGCG) — expected from double-stranded DNA

**Convergence:** Converged in 1881 iterations, reconstruction error = 0.1423.

---

### 6.2 — Autoencoder (Neural Network, 2D bottleneck)

**What it does:** A neural network that compresses 256D → 2D → 256D. The bottleneck forces the network to learn the most important 2D representation to reconstruct the input.

Architecture: `256 → 64 → 16 → 2 → 16 → 64 → 256`, trained for 200 epochs with MSE loss.

**Result: Bottleneck collapsed.** Both output dimensions have near-zero variance (0.0004, 0.0023) — all 314 MAGs mapped to essentially one point. The network found a trivial solution by epoch 20 and stopped learning.

**Why:** 314 samples is too small for a neural network with thousands of parameters. It memorized reconstruction without needing to use the bottleneck meaningfully.

**Lesson:** Deep learning is not always better. At N=314, purpose-built methods (UMAP, NMF) outperform neural networks.

---

### 6.3 — UMAP Hyperparameter Sweep

**Motivation:** UMAP has two key parameters that dramatically affect the output:
- `n_neighbors` — how many nearby points define a neighborhood (local vs global structure)
- `min_dist` — how tightly points can pack in 2D

Rather than using default values, we systematically tried **12 combinations** (4 × 3 grid) and picked the one with the best cluster separation (silhouette score).

**Winner:** `n_neighbors=15, min_dist=0.0` with silhouette = **0.678**. This is the UMAP embedding used for all downstream analysis.

---

### 6.4 — Semi-supervised UMAP

**What it does:** The same UMAP, but told the family labels for the 126 labeled MAGs during fitting. Unlabeled MAGs are marked -1 and treated as unknown. The embedding is pulled toward grouping same-family MAGs together.

**What the plot showed:**
- Same-family MAGs form very tight, well-separated clusters
- The isolated brown cluster (bottom-left) is a phylogenetically distinct family far from all others
- Unlabeled MAGs follow the same overall structure, suggesting they belong to the same families

**Why it's not a fair benchmark:** Evaluating taxonomy alignment on an embedding that used taxonomy labels to build it is circular. This is an upper-bound visualization only — not included in the fair comparison.

---

### 6.5 — Agglomerative Clustering (Cosine Distance + Average Linkage)

**What it does:** Builds a tree of merges from the bottom up — starts with every MAG as its own cluster, repeatedly merges the two closest clusters until everything is one group. The tree (dendrogram) is then cut at a height to get flat clusters.

Uses **cosine distance** between k-mer frequency vectors — measures the *angle* between two vectors rather than their absolute distance. Better for compositional data (frequencies) than Euclidean distance.

**Result:**
- Best k=2, silhouette = **0.547**
- The dendrogram shows one massive gap at height ~0.19 — the two groups are genuinely distinct
- Orange cluster (left): tight, low merging distances → homogeneous low-GC genomes
- Green cluster (right): more diverse internal structure → high-GC genomes with varying compositions
- Silhouette drops monotonically for k=3 through k=10 — no finer natural structure found this way

**Interpretation:** Cosine agglomerative clustering recovers the same dominant GC-content split seen in PCA and UMAP. It confirms this two-group structure is real and not an artifact — but it's too coarse to recover family-level taxonomy.

---

### 6.6 — HDBSCAN on Tuned UMAP

**What it does:** A density-based clustering algorithm — finds groups by looking for regions where points are packed tightly, rather than assuming spherical clusters. Points in sparse regions are labeled **noise (-1)** instead of forced into a cluster. Run on the best UMAP embedding from 6.3.

**Why on UMAP first:** HDBSCAN struggles in high dimensions. UMAP compresses 256D → 2D while preserving cluster structure; HDBSCAN then finds dense regions in that clean 2D layout.

**Result:**
- **13 clusters, 3 noise points (0.96%), silhouette = 0.6445**
- Cluster sizes vary from 10 to 52 MAGs — reflecting natural variation in how many MAGs belong to each group
- The isolated cluster (far left, 12 MAGs) matches the *Porphyromonas* group identified in Notebook 02
- Only 3 noise points — almost the entire dataset has clear cluster membership

---

## Section 7 — Final Evaluation: Which Method Actually Works?

**Motivation:** High silhouette score means clean geometric clusters — but it doesn't mean the clusters match biology. We evaluate every method on four separate criteria.

### Results summary

| Method | Internal Sil. | Family Sil. | AMI | Mean Purity |
|---|---|---|---|---|
| **NMF-10d (k=10)** | 0.380 | **0.274** | **0.648** | **0.778** |
| PCA-10d (k=2) | 0.416 | 0.224 | 0.341 | 0.455 |
| Agglomerative (k=2) | 0.547 | 0.243 | — | — |
| UMAP+HDBSCAN (k=13) | 0.644 | — | — | — |
| UMAP-tuned k-means (k=5) | 0.723 | 0.138 | 0.519 | 0.739 |
| UMAP-semisup k-means (k=3) | **0.734** | 0.120 | 0.330 | 0.555 |
| Autoencoder (k=2) | 0.651 | -0.197 | 0.341 | 0.455 |

**AMI** = Adjusted Mutual Information — how well cluster assignments match family labels, corrected for random chance. 0 = no better than random, 1 = perfect match.

**Key finding — the disconnect:** UMAP methods have the highest internal silhouette (geometrically clean) but low family silhouette. **NMF has the lowest internal silhouette but wins every biological metric.** High internal silhouette ≠ biological accuracy.

**NMF cluster purity:** Six of ten NMF clusters reached **purity = 1.0** — every labeled MAG inside those clusters belongs to the same bacterial family (Bacteroidaceae, Porphyromonadaceae, Campylobacteraceae, Brachyspiraceae). No other method achieves this.

**Autoencoder confirmed as equivalent to PCA:** ARI between autoencoder and PCA-10d = 0.97. Both found the same trivial k=2 split. The collapsed bottleneck produced meaningless results confirmed by negative family silhouette (-0.197).

**No sample confound:** All methods have sample ARI < 0.06. None of the clustering reflects which sequencing sample a MAG came from — the signal is biology, not batch effects.

**Cross-method groups (ARI heatmap):**
- PCA + Autoencoder + Agglomerative (ARI 0.95–0.97): same coarse GC split (k=2)
- NMF + UMAP+HDBSCAN (ARI=0.56): both find finer structure beyond GC
- UMAP-semisup disagrees with everything (ARI 0.08–0.34): label-guided, not comparable

---

## Section 8 — Overall Conclusions

### Hypotheses — final verdict

| Hypothesis | Result | Evidence |
|---|---|---|
| **H1** — disease phase separable | ❌ Not supported | Phase silhouette ≈ 0 across all embeddings; acute/recovery MAGs interleave |
| **H2** — taxonomy in k-mer space | ✅ Supported | Extreme-GC phyla separate cleanly; *Porphyromonas* isolated without labels; NMF AMI=0.648 |
| **H3** — GC content co-varies | ✅ Strongly supported | PC1 = GC axis; UMAP global split = GC split; phyla sort by GC |
| **H4** — 4-mers better than 2-mers | ❌ Rejected | k=2 silhouette (0.275) > k=4 (0.237) on raw features |

### What worked and what didn't

**Worked:**
- NMF-10d — best biological clustering, interpretable components, family-level purity
- UMAP + HDBSCAN — best geometric clustering, fine-grained 13 clusters, biologically promising
- Semi-supervised UMAP — confirms taxonomy is strongly encoded in k-mer space

**Didn't work:**
- Autoencoder — bottleneck collapsed at N=314; deep learning needs more data
- k-mer length sweep — adding more dimensions doesn't automatically help; method matters more than feature size

### Core takeaway

> 4-mer k-mer composition encodes bacterial taxonomy at the phylum and family level. The dominant signal is GC content (H3 ✅), which correctly separates major phyla. Within moderate-GC phyla, NMF's parts-based decomposition recovers family-level structure better than any other method tested — achieving 0.648 AMI and 0.778 mean cluster purity on the 126 labeled MAGs. Disease phase succession is not recoverable from k-mer composition alone (H1 ❌), suggesting that microbial succession during cholera recovery is driven by *which species* return, not by changes within individual genomes.

---

## Appendix — Method Quick Reference

| Term | Plain English |
|---|---|
| MAG | A partial bacterial genome reconstructed from mixed environmental DNA |
| k-mer | A DNA substring of fixed length k (e.g. AAAC is a 4-mer) |
| GC content | Fraction of the genome that is G or C bases (vs A or T) |
| Silhouette score | How well-separated clusters are: 1=perfect, 0=overlapping, -1=wrong |
| AMI | How much cluster labels agree with taxonomy labels, corrected for chance |
| Purity | Fraction of labeled MAGs in a cluster that belong to the dominant family |
| Taxonomy | Classification of organisms: Domain → Phylum → Class → Order → Family → Genus → Species |
| Phylum | A broad group of related bacteria, e.g. Bacteroidota, Proteobacteria |
| GTDB-Tk | Software that assigns taxonomy to MAGs by comparing to a reference database |
