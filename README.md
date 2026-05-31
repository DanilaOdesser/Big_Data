
# Big Data Final Project — DNA Sequence Clustering

## Overview

Cluster microbial genomes (MAGs) by sequence similarity/dissimilarity using embedding and dimensionality-reduction methods from the course.

- **Unit of observation**: one MAG (Metagenome-Assembled Genome) = one `.fa.gz` file
- **Current dataset**: David_2015_Bangladesh — **314 MAGs** in `data/raw_David_2015_bangladesh/`
- **Data format**: FASTA (`.fa.gz`) — each file contains multiple contigs

---

## Data Format

Each `.fa.gz` is a gzip-compressed FASTA file with one or more contigs:

```
>k119_1729
GTGGCACAGGGAACCCGCCACAGCAT...
>k119_3850
ACGTTAGCCA...
```

Tab-delimited conversion (via `seqkit`):
```
k119_1729    GTGGCACAGGG...
k119_3850    ACGTTAGCCA...
```

Example contig sizes from one file (15 contigs, ~230 kb total):
```
k119_1729 →  2,707 bp    k119_3850 → 10,731 bp
k119_399  → 40,662 bp    k119_3298 → 22,490 bp
k119_3683 → 20,156 bp    k119_4313 →  5,935 bp
...
```

---

## Feature Engineering (Multidimensional Representation)

Each genome becomes one row; columns are features derived from DNA:

| Feature group         | Dimensions | Notes                                      |
|-----------------------|------------|--------------------------------------------|
| 4-mer frequencies     | 256        | Main approach — captures local sequence patterns |
| 3-mer frequencies     | 64         | Smaller, faster alternative                |
| GC content            | 1          | Simple global statistic                    |
| Total genome size     | 1          | Sum of all contig lengths                  |
| Number of contigs     | 1          | Assembly fragmentation proxy               |
| Contig length stats   | ~4         | mean, std, min, max                        |

**Recommended starting point**: 4-mer frequencies (256-dim) per genome → apply PCA/UMAP/t-SNE.

---

## Dataset Background & Scientific Context

### Original Study

**David et al. 2015** — *"Gut Microbial Succession Follows Acute Secretory Diarrhea in Humans"*
- **Journal:** mBio 6(3):e00381-15 | **DOI:** [10.1128/mBio.00381-15](https://doi.org/10.1128/mBio.00381-15) | **PMID:** [25991682](https://pubmed.ncbi.nlm.nih.gov/25991682/)
- **ENA accession:** [PRJEB9150](https://www.ebi.ac.uk/ena/browser/view/PRJEB9150)
- **Where:** ICDDR,B Dhaka Hospital, Bangladesh (world's leading diarrhoeal disease research centre)
- **What:** Longitudinal fecal sampling of ~40 cholera/ETEC patients at days 0, 1, 7, 30 post-infection
- **Methods:** 16S rRNA sequencing (114 samples) + deep shotgun metagenomics (47 samples) — the shotgun samples are what SPIRE re-processed into the 314 MAGs we have

**Key biological findings:**
- Gut microbiome recovery after cholera follows **deterministic ecological succession** (not random) — comparable to pioneer-climax community assembly in macroecology
- Bacteriophage abundance spikes **88.9x** at acute infection, suggesting phages drive dysbiosis
- Same successional pattern holds in ETEC patients → universal feature of diarrhoea recovery
- *Prevotella*-dominated baseline (typical of plant-based diets) is disrupted and then restored

### The SPIRE Database

**Schmidt, Fullam et al. 2024** — *"SPIRE: a Searchable, Planetary-scale mIcrobiome REsource"*
- **Journal:** Nucleic Acids Research 52(D1):D777–D783 | **DOI:** [10.1093/nar/gkad943](https://doi.org/10.1093/nar/gkad943) | **Web portal:** [spire.embl.de](https://spire.embl.de/)
- **Institution:** EMBL Heidelberg (Bork group)
- **Scale:** 99,146 metagenomic samples from 739 studies → **1.16 million quality-filtered MAGs**
- **Pipeline:** QC trimming → human DNA removal → assembly (megahit) → binning (metaBAT2) → quality (CheckM2 ≥50% completeness, ≤10% contamination) → taxonomy (GTDB-Tk)
- **Our data:** David_2015_Bangladesh = 47 shotgun samples re-processed → 314 MAGs

### Related Studies Using This Cohort

| Study | Key finding relevant to clustering |
|---|---|
| [Hsiao et al. 2014, *Nature*](https://doi.org/10.1038/nature13738) | *Ruminococcus obeum* restricts *V. cholerae* via quorum sensing — identifies key recovery-phase taxa |
| [Midani & Weil et al. 2018, *J. Infect. Dis.*](https://doi.org/10.1093/infdis/jiy192) | Baseline microbiota predicts cholera susceptibility (AUC=0.80); *Prevotella*-depleted = susceptible |
| [Alavi et al. 2020, *Cell*](https://doi.org/10.1016/j.cell.2020.05.036) | Reanalysed same metagenomes; bile-salt hydrolase genes deplete during acute cholera — links MAG function to disease |
| [Levade, Weil et al. 2021, *J. Infect. Dis.*](https://doi.org/10.1093/infdis/jiaa358) | Shotgun metagenomics outperforms 16S for outcome prediction |

### Hypotheses for This Project

The 314 MAGs represent microbial genomes recovered from acute (day 0–1) and recovery (day 7–30) stool samples. Based on the literature, these are testable hypotheses:

**H1 — Succession clusters (primary hypothesis)**
> MAGs from the *acute* phase and the *recovery* phase form distinct clusters in k-mer embedding space, reflecting the known ecological succession of the microbiome.
> *Prediction:* t-SNE/UMAP should separate early-infection genomes (facultative anaerobes, Proteobacteria) from late-recovery genomes (strict anaerobes: Bacteroides, Prevotella, Ruminococcus).

**H2 — Taxonomic phylogeny reflected in sequence space**
> Genomes of the same phylum/class cluster together in k-mer frequency space even without using taxonomy labels.
> *Prediction:* Silhouette score computed against GTDB-Tk taxonomy labels should be significantly > 0 for 4-mer embeddings.

**H3 — GC content and genome size co-vary with taxonomy**
> Different microbial lineages have characteristic GC content and genome size; these simple features should partially recapitulate k-mer clustering.
> *Prediction:* PCA on GC + size alone gives weaker clusters than 4-mer PCA, but the axes should correlate.

**H4 — Dimensionality reduction quality decreases with fewer k-mer dimensions**
> 4-mers (256 dim) preserve more biological signal than 2-mers (16 dim) or 3-mers (64 dim).
> *Prediction:* Silhouette score is monotonically higher as k increases from 2 → 4.

---

## Project Requirements

1. **Hypothesis**: Define a clear hypothesis about the data structure (e.g., "Genomes from the same host cluster together in k-mer space")
2. **Embedding methods**: Apply several methods from the course, e.g.:
   - PCA
   - t-SNE
   - UMAP
   - Autoencoders (optional, more complex)
3. **Evaluation**: Compare embeddings using quantitative metrics, e.g. **Silhouette index**
4. **Clustering**: Recover cluster criteria/cutoffs (e.g., k-means, DBSCAN, hierarchical)

---

## Available Datasets

| Dataset | Host | Size | MD5 |
|---|---|---|---|
| `David_2015_Bangladesh` | Human (Bangladesh) | 137 MB | `58e2dd4c7d8e57c3af5b9ff3b0d6fa10` |
| `Devoto_2019_Bangladeshi` | Human (Bangladesh) | 176 MB | `0e4361d8e264854ae26feb4c037eb766` |
| `Rampelli_2015_Hadza` | Human (Hadza hunter-gatherers) | 384 MB | `e801d2660ffa3a2083060a11caa7a52a` |
| `Tung_2015_baboon` | Baboon | 788 MB | `4c8356d304f24844328c3c585a4df36c` |
| `Andersen_2019_pig` | Pig | 9.7 GB | `73b087d761cbad8261eb5044c9a5c7b7` |

Download base URL: `https://swifter.embl.de/~fullam/spire/compiled/<dataset_name>.tar`

**Current local data**: `David_2015_Bangladesh` (smallest, 314 MAGs) — good for prototyping.

---

## Project Structure

```
Big_Data/
├── data/
│   └── raw_David_2015_bangladesh/   # 314 .fa.gz files (MAGs)
├── models/
├── sample.ipynb                     # Starter notebook
├── requirements.txt
└── README.md
```

---

## Useful Tools

- [`seqkit`](https://bioinf.shenwei.me/seqkit/) — FASTA parsing, format conversion, stats
- `BioPython` (`Bio.SeqIO`) — Python FASTA parsing
- `scikit-learn` — PCA, t-SNE, k-means, Silhouette score
- `umap-learn` — UMAP embeddings

---

## Course Algorithms (by Lecture)

Course: *Embeddings: From Theory to Practice* — Constructor University (Tupikina & Kukushkin)

---

### Lecture 1 — Dimensionality Reduction Foundations

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **PCA** | Linear | Maximize variance via eigenvectors of covariance matrix | target dim `q` |
| **MDS** | Non-linear | Minimize stress = sum of (d_orig − d_embed)² | target dim `q` |
| **Local MDS** | Non-linear | MDS + repulsion term for K-nearest neighbors | K neighbors |
| **ISOMAP** | Manifold | Build K-NN graph → geodesic distances → MDS | K or ε radius |

---

### Lecture 2 — t-SNE

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **t-SNE** | Non-linear | High-d Gaussian → low-d Student-t; minimizes KL divergence | perplexity, learning rate, n_iter |

Barnes-Hut approximation reduces complexity from O(N²) to O(N log N). No out-of-sample transform; non-deterministic.

---

### Lecture 3 — SVD, NMF, Collaborative Filtering

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **Cosine CF** | RecSys (kNN) | Similarity = cosine over shared ratings; find K nearest users/items | K neighbors, M min co-ratings |
| **SVD** | Matrix factorization | X = U D Vᵀ; truncate to rank q (Eckart-Young optimal) | rank `q` |
| **Funk SVD** | Matrix factorization | Learns P, Q via SGD; objective = MSE + L2 regularization | n_factors, lr η, reg λ, n_epochs |
| **SVD++** | Matrix factorization | Funk SVD + user/item bias + implicit feedback vectors | + biases, implicit y_k |
| **NMF** | Matrix factorization | X ≈ WH with W, H ≥ 0; Lee-Seung multiplicative updates | rank `k` |

Evaluation: RMSE on observed entries, 2D latent-space visualization.

---

### Lecture 4 — Neural Embeddings & Autoencoders

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **Word2Vec / \*2vec** | Neural embedding | Skip-gram / CBOW; negative sampling or hierarchical softmax | embedding dim, window size |
| **Autoencoder** | Neural dim. reduction | Encoder-decoder bottleneck; minimizes reconstruction loss | latent dim, architecture |
| **Variational AE (VAE)** | Generative | Probabilistic latent space; KL + reconstruction loss | latent dim, β weight |

Node2vec and Doc2vec follow the same \*2vec family principle.

---

### Lecture 5 — UMAP

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **UMAP** | Manifold | Fuzzy topological graph in high-d; cross-entropy loss alignment in low-d | `n_neighbors`, `min_dist`, metric |

Advantages over t-SNE: faster, preserves global structure, supports out-of-sample transform, supervised mode (`fit_transform(X, y=labels)`).  
Metrics: Euclidean (images/k-mers), cosine (sparse/text).

---

### Lecture 6 — Markov Models & HMMs

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **Markov Chains** | Probabilistic sequence | Row-stochastic transition matrix A; stationary dist π* = π*A | K states |
| **PageRank** | Graph ranking | Markov chain on web graph with teleportation; d ≈ 0.85 | damping factor d, iterations |
| **HMM — Forward** | Sequence evaluation | Compute P(observations \| model) in O(N²T) | N states, T timesteps |
| **HMM — Viterbi** | Sequence decoding | Most probable hidden state path via DP in O(N²T) | N states, T timesteps |
| **HMM — Baum-Welch** | Sequence learning | EM algorithm to estimate A, B, π from observations | convergence threshold |

Applications: gene finding, speech recognition, market regime detection, user behavior modeling.

---

### Lecture 7 — Multiple Sequence Alignment

| Algorithm | Type | Key idea | Parameters |
|---|---|---|---|
| **Needleman-Wunsch** | Global alignment | Dynamic programming on full sequence pairs; traceback | match +m, mismatch −x, gap −g |
| **Progressive MSA** | Multiple alignment | Build guide tree → merge profiles bottom-up | profile scoring function |
| **Agglomerative clustering** | Clustering | Merge closest sequences; distance = q-gram cosine | q (sub-sequence length) |
| **Sequen-C** | Visualization | Information score I = 1 − entropy/log₂(\|A\|+1); prune low-info columns | threshold I_τ |

---

### Algorithm Quick-Reference for This Project

Most relevant algorithms for DNA/MAG clustering:

| Priority | Algorithm | Why useful here |
|---|---|---|
| 1 | **PCA** | Fast baseline; interpretable axes; good for 256-dim k-mer vectors |
| 2 | **UMAP** | Best for large datasets; preserves global structure; fast |
| 3 | **t-SNE** | Strong local clustering; good for visualization |
| 4 | **Agglomerative clustering** | Natural fit; no need to fix K in advance |
| 5 | **k-means + Silhouette** | Quantitative evaluation of cluster quality |
| 6 | **Autoencoder** | Non-linear compression if k-mer space is complex |
| 7 | **Needleman-Wunsch** | Direct sequence alignment (computationally expensive at scale) |

**Evaluation metrics used in course:** RMSE, Silhouette index, kNN accuracy, information score, PageRank convergence.

---

## Course Resources

- Course repo: https://github.com/Liyubov/course_embeddings_constructor
- Notebooks in repo: `03_funk_svd.ipynb` (SVD/RecSys), `06_markov.ipynb` (PageRank/HMM), `05_umap_embeddings.ipynb` (UMAP, in PR)


## Useful Links

- Contrastive learning and Dmitry Kobak lecture: https://www.youtube.com/watch?v=A2HmdO8cApw
- All biomedical researches in the world data: https://static.nomic.ai/pubmed
- t-SNE for Biology: https://www.nature.com/articles/s41467-019-13056-x
- Parametric UMAP: https://umap-learn.readthedocs.io/en/latest/parametric_umap.html
- TODO: https://www.youtube.com/watch?v=qJeaCHQ1k2w