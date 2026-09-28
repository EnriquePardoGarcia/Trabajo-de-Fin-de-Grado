# Sparse LISTA Autoencoders for Mitotic Figure Representation in H&E Histology

> **Bachelor's Thesis (TFG)** — Grado en Ciencia e Ingeniería de Datos, Universidade da Coruña (UDC)
> **Title:** *Representación y análisis de células mitóticas en histología H&E mediante autoencoders dispersos basados en LISTA*
> **Subtitle:** *Study of the effect of the loss function and the dictionary size on unsupervised sparse representations*
> **Author:** Enrique Pardo García
> **Supervisors:** José Rouco Maseda · Jorge Novo Buján (VARPA group, UDC)

---

## Table of contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Key ideas at a glance](#key-ideas-at-a-glance)
4. [Background](#background)
5. [Method](#method)
6. [Dataset](#dataset)
7. [Experimental design](#experimental-design)
8. [Main results](#main-results)
9. [Discussion](#discussion)
10. [Limitations and future work](#limitations-and-future-work)
11. [Getting started](#getting-started)
12. [Glossary](#glossary)
13. [References](#references)
14. [Citation](#citation)
15. [Acknowledgements and contact](#acknowledgements-and-contact)

---

## Overview

This project studies whether an **interpretable, label-free representation** of histology patches can be learned with a **sparse autoencoder whose encoder is an unrolled sparse-coding solver (LISTA)**, and how that representation behaves when two design choices change:

- the **reconstruction loss** (MSE, MAE or SSIM), and
- the **dictionary size** ($m = 128$ vs. $m = 1024$ atoms).

The model is trained **without any labels** on H&E-stained patches from the MIDOG dataset. Labels (mitotic / non-mitotic) are used **only at evaluation time**, to measure whether the learned sparse codes carry any discriminative signal.

The thesis is therefore both a **representation-learning** study (can we obtain a sparse, interpretable code that reconstructs cells faithfully?) and an **analysis** study (why does training sometimes collapse, and what does the code encode?).

## Motivation

Automatic mitosis detection is clinically relevant because the **mitotic index** is a key prognostic marker of tumour aggressiveness (e.g. in Elston–Ellis grading). In practice:

- Counting mitoses on whole-slide images (WSIs) is **slow and tedious**, with **high inter-observer variability** (around 20 % disagreement between experts in MIDOG).
- Supervised automatic systems need **large amounts of expert-annotated data**, which is expensive.
- Most deep models are **black boxes**: dense representations that cannot be audited or explained.

These two problems (costly labels and lack of interpretability) motivate the central research question:

> **How can we obtain an interpretable system that separates mitotic from non-mitotic figures, without depending on labels?**

The proposed answer explores **sparse coding** (interpretability by construction), the **loss function** and the **dictionary size**.

## Key ideas at a glance

| Idea | One-line summary |
|---|---|
| Sparse coding | Each patch is explained by a *few* active atoms of a learned dictionary. |
| LISTA | A fixed number of unrolled ISTA iterations turned into a trainable network ($T = 3$ steps). |
| Tied weights | The decoder is $\mathbf{W}_e^\top$, so an atom that is *detected* is the same one used to *reconstruct*. |
| Adaptive sparsity | A controller adjusts the L1 weight $\lambda_{L1}$ during training to keep the code sparse but not collapsed. |
| Label-free training | Labels are never used to train, only to evaluate. |
| Main finding | Dictionary size alone does not predict stability; the **loss function** does. |

## Background

### Sparse coding

Given an input $\mathbf{x}\in\mathbb{R}^n$ and a dictionary whose atoms are patterns, we look for a code $\mathbf{z}\in\mathbb{R}^m$ with mostly zeros such that $\mathbf{x}\approx$ a weighted sum of a few atoms:

$$
\mathbf{z}^\ast = \arg\min_{\mathbf{z}}\ \tfrac{1}{2}\lVert \mathbf{x}-\mathbf{W}_d\mathbf{z}\rVert_2^2 + \lambda\,\lVert\mathbf{z}\rVert_1 .
$$

- The **$\ell_0$ "norm"** (count of non-zeros) would give true sparsity, but the resulting problem is **NP-hard**.
- It is **relaxed to $\ell_1$**, which is convex and tractable. Geometrically, the $\ell_1$ ball has *vertices on the coordinate axes*, so the optimum tends to land where many coordinates are exactly zero (the same argument that underlies LASSO).

### ISTA / FISTA

**ISTA** solves the problem iteratively with a gradient step followed by *soft-thresholding*:

$$
\mathbf{z}^{(k+1)} = \mathcal{S}_{\lambda/L}\!\left(\mathbf{z}^{(k)} - \tfrac{1}{L}\mathbf{W}_d^\top(\mathbf{W}_d\mathbf{z}^{(k)}-\mathbf{x})\right),
\qquad \mathcal{S}_\theta(v)=\mathrm{sign}(v)\max(|v|-\theta,0).
$$

**FISTA** adds momentum for faster convergence. Both share two limitations:

1. They are **content-agnostic**: every new image is optimised from scratch, requiring many iterations.
2. Learning the dictionary needs **alternating optimisation** (infer codes, then update the dictionary).

### LISTA

**LISTA** (*Learned ISTA*) unrolls a **fixed number $T$ of ISTA iterations** and makes their matrices **trainable parameters**. It addresses both limitations at once: inference takes a small, fixed number of steps (on the order of 18–35× fewer iterations than FISTA for comparable error), and the dictionary is learned end-to-end by gradient descent.

## Method

### Architecture

Notation: $n = 64\times64\times3 = 12{,}288$ (flattened RGB patch), $m\in\{128,1024\}$ (number of atoms).

| Symbol | Shape | Role |
|---|---|---|
| $\mathbf{x}$ | $n\times1$ | Input patch (flattened) |
| $\mathbf{W}_e$ | $m\times n$ | Dictionary / projection. **Each row is an atom** (a visual pattern of size $n$). |
| $\mathbf{S}$ | $m\times m$ | Lateral matrix: lets atoms inhibit / compete with each other |
| $\mathbf{b}_S$ | $m\times1$ | Per-atom threshold (bias) |
| $\mathbf{z}$ | $m\times1$ | Sparse code (mostly zeros) |
| $\hat{\mathbf{x}}$ | $n\times1$ | Reconstruction |

**Encoder (unrolled LISTA, $T=3$):**

$$
\mathbf{B}=\mathbf{W}_e\mathbf{x}
$$
$$
\mathbf{z}^{(1)}=\mathrm{ReLU}(\mathbf{B}+\mathbf{b}_S),\qquad
\mathbf{z}^{(t+1)}=\mathrm{ReLU}(\mathbf{B}+\mathbf{S}\,\mathbf{z}^{(t)}+\mathbf{b}_S)
$$

The projection $\mathbf{B}$ is computed **once** and reused. The ReLU produces exact zeros (sparsity), and $\mathbf{S}$ suppresses redundant atoms that would otherwise "explain the same thing twice".

**Decoder (tied weights):**

$$
\hat{\mathbf{x}}=\mathbf{W}_e^\top\mathbf{z}=\sum_{i:\,z_i\neq0} z_i\,\mathbf{w}_i .
$$

The reconstruction is a **linear combination of a few active atoms**, which is precisely what makes the code interpretable: for any patch you can list *which* patterns were used and *with what weight*.

> Since $m \ll n$, the dictionary is **undercomplete** in size; sparsity adds a second restriction on top of it (few active atoms among those available).

### What is learned

Only three sets of parameters, all trained by gradient descent and **without labels**: $\mathbf{W}_e$ (which patterns exist), $\mathbf{S}$ (how atoms compete) and $\mathbf{b}_S$ (how easy each atom is to activate). $T$, the architecture and the loss are design choices, and $\lambda_{L1}$ is handled by a controller rather than by backpropagation.

### Loss functions

The reconstruction term can be instantiated with:

- **MSE** — squared pixel error; smooth gradients, sensitive to outliers.
- **MAE** — absolute pixel error; more robust to outliers, constant-magnitude gradient.
- **SSIM** — compares *local windows* through luminance, contrast and structure, so it rewards preserved structure rather than raw intensity agreement.

A **centred Gaussian spatial mask** weights the reconstruction error so that the cell (usually at the centre of the patch) matters more than the background, without requiring any segmentation.

### The sparsity–reconstruction dynamic and dead atoms

Training is a system with **two opposing forces**:

- learning a good dictionary tends to **activate more atoms**;
- sparsifying the code tends to **activate fewer atoms**.

$\lambda_{L1}$ arbitrates between them. If nothing regulates it, the code either **collapses** (0 % active atoms, empty reconstruction) or **saturates** (almost everything active). A related failure is the **dead atom**: an atom that is never activated and therefore stops learning.

### Control mechanisms

1. **Automatic $\lambda_{L1}$ adjustment** — a controller raises or lowers $\lambda_{L1}$ (multiplicatively, smoothed with an exponential moving average) according to the fraction of active atoms versus a target (≈ 50 %).
2. **Revival** — atoms detected as dead are re-oriented towards poorly reconstructed inputs instead of being discarded (the dictionary size stays fixed).
3. **Dead-atom detection** — identifies inactive atoms before intervening.

### Training schedule

| Phase | What changes | Purpose |
|---|---|---|
| **Warm-up** | $\mathbf{S}$ frozen, $\lambda_{L1}=0$ | Let the dictionary settle by reconstruction alone |
| **Phase A** | $\mathbf{S}$ free, still no sparsity pressure | Learn lateral competition on a stable dictionary |
| **Phase B** | Controller adjusts $\lambda_{L1}$ | Introduce and regulate sparsity dynamically |

### Checkpoints

Three kinds of checkpoint are stored: **best**, **last** and **jump** (saved when the tracked metric varies by more than 25 % between consecutive epochs). Comparing only *best* vs *last* can hide collapses or reactivations that happen mid-training; the jump checkpoints preserve the **full trajectory**.

### Pipeline

1. **Training** (unlabelled): standardisation + augmentation → LISTA encoder → tied decoder → loss (reconstruction + sparsity).
2. **Inference**: weights frozen; the encoder simply computes $\mathbf{z}$ for each *un-augmented* test patch.
3. **Evaluation** (labels used here for the first and only time): reconstruction, sparsity, dictionary coherence and discriminability.

## Dataset

**MIDOG** (Mitosis Domain Generalization): H&E-stained histology with mitotic-figure annotations, digitised with **several different scanners** (and covering multiple species). H&E is the standard stain: haematoxylin (blue/violet) marks nuclei, where mitotic chromatin condensation is visible, and eosin (pink) marks cytoplasm and matrix.

Why MIDOG:

- **Multi-scanner / multi-species** → a domain-shift-aware reference, which discourages the dictionary from learning scanner-specific colour or noise shortcuts instead of real morphology.
- Annotated specifically for **mitosis**, which fits the research question.

Usage in this work:

- Patches of **64 × 64 px, 3 channels** ($n = 12{,}288$).
- **Unlabelled** patches for training (≈ 21,000), **labelled** patches for test-time evaluation.
- Standardisation and augmentation during training help attenuate scanner and stain variability.

## Experimental design

Three losses × two dictionary sizes = **six configurations**, named `<loss>_aug_<m>` (e.g. `ssim_aug_1024`). For readability they are grouped into three blocks:

| Block | Configurations | Description |
|---|---|---|
| **Dim 128** | `mse/mae/ssim_aug_128` | Small-dictionary reference |
| **Dim 1024 pointwise** | `mse/mae_aug_1024` | Large dictionary with pixel-wise losses |
| **Dim 1024 SSIM** | `ssim_aug_1024` | Large dictionary with structural loss |

### Evaluation metrics

- **Reconstruction:** MSE, MAE, PSNR, SSIM.
- **Sparsity:** mean $\ell_0$ / fraction of zeros.
- **Dictionary coherence** $\mu(\mathbf{W}_e)$: how close to parallel the atoms are ($\approx0$ = nearly orthogonal, $\approx1$ = redundant).
- **Discriminability** of each atom w.r.t. the label: **Fisher index**, **ANOVA F-score** and **mutual information**. Codes are also projected with **PCA** and **t-SNE**.

## Main results

| Block | Training behaviour | Active atoms | Mean coherence | Reconstruction |
|---|---|---|---|---|
| **Dim 128** | Stable | ≈ 50 % | ≈ 0.01 (almost orthogonal) | Faithful |
| **Dim 1024 pointwise (MSE/MAE)** | **Collapse** | 0 % | ≈ 0.31 (peak ≈ 0.8, very redundant) | Uniform colour ($\hat{\mathbf{x}}\approx\mathbf{0}$) |
| **Dim 1024 SSIM** | Stable, no collapse | ≈ 16.5 % | ≈ 0.16 (no peak) | Structure preserved |

Key observations:

- **Dictionary size alone does not predict stability.** With the same $m = 1024$, pointwise losses collapse while SSIM does not — a qualitative **bifurcation**, not a gradual degradation.
- **Coherence, reconstruction, sparsity and the reconstruction–sparsity trade-off tell the same story** — they are different views of the same phenomenon.
- **Criteria agreement is lost where the code collapses.** In Dim 128 Fisher and ANOVA-F agree on the top atoms; in the collapsed block they diverge; in Dim 1024 SSIM only mutual information remains stable.
- **The discriminative signal is weak but real.** Atom-level statistics show some structure, but neither PCA nor t-SNE reveals class-separable clusters (not even in the most favourable case, `ssim_aug_1024`).

## Discussion

- **Why does the code collapse at $m=1024$?** The $\lambda_{L1}$ controller was calibrated on $m=128$; its seed and growth rate are too conservative for a dictionary eight times larger, so it never raises $\lambda_{L1}$ enough to escape the degenerate fixed point $\hat{\mathbf{x}}=\mathbf{0}$. The problem is one of **calibration**, not of dimensionality per se.
- **Is SSIM intrinsically better?** Not necessarily: it may simply start with a fraction of active atoms already close to the target and therefore never fall into the collapse.
- **Why the trajectories matter.** Only the full training trajectory (jump checkpoints) exposes this; comparing best vs last would hide it.

## Limitations and future work

| Limitation | Future work |
|---|---|
| The $\lambda_{L1}$ controller is calibrated only for $m=128$ (fixed seed and growth rate) | Recalibrate as a function of $m$ (scaled seed and growth) |
| Only two dictionary sizes, chosen as extremes | Explore intermediate sizes to locate the bifurcation point and draw the full reconstruction-vs-$m$ curve |
| Discriminative signal is validated only with univariate statistics (Fisher, ANOVA-F, MI) | Supervised fine-tuning: a light classifier (e.g. logistic regression) on $\mathbf{z}$, reporting AUC / F1 |
| A single dataset (MIDOG) | Validate on other histological datasets, scanners and tissues |
| No multi-seed variance study | Repeat each configuration with several seeds |

Further ideas: hybrid losses (e.g. MSE + SSIM), an active deduplication mechanism for near-parallel atoms, and comparison against a dense autoencoder trained under the same protocol.

## Getting started

> **Note:** adapt this section to the actual repository. The commands below are placeholders.

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-name>

# 2. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Prepare the data (MIDOG patches, 64x64 RGB)
#    <describe where the data must be placed / how to preprocess it>

# 4. Train one configuration (example name: ssim_aug_1024)
python train.py --loss ssim --dict-size 1024 --augment
#    <replace with the real entry point and arguments>

# 5. Evaluate a checkpoint (best / last / jump)
python evaluate.py --config ssim_aug_1024 --checkpoint best
```

Suggested layout (edit to match the real project):

```text
.
├── README.md
├── src/            # model (LISTA encoder, tied decoder), losses, controller, training loop
├── configs/        # one config per experiment, e.g. ssim_aug_1024
├── notebooks/      # analysis and figures
├── results/        # metrics, checkpoints, figures per configuration
└── docs/           # thesis report and defence slides
```

## Glossary

- **Atom:** one pattern (row of $\mathbf{W}_e$); the reconstruction is a weighted sum of active atoms.
- **Dictionary:** the collection of $m$ atoms, $\mathbf{W}_e\in\mathbb{R}^{m\times n}$.
- **Sparse code $\mathbf{z}$:** the vector of coefficients, mostly zero.
- **LISTA:** Learned ISTA — unrolled ISTA iterations with trainable parameters.
- **Tied weights:** the decoder reuses the encoder matrix (transposed).
- **$\lambda_{L1}$:** weight of the sparsity penalty, managed by the controller.
- **Dead atom:** an atom that is (almost) never activated and stops learning.
- **Revival:** re-orienting dead atoms towards poorly reconstructed inputs.
- **Coherence:** maximum/mean normalised inner product between distinct atoms.
- **Collapse:** the code goes to zero active atoms and the reconstruction becomes empty.
- **WSI:** whole-slide image (a fully digitised slide).
- **H&E:** haematoxylin and eosin staining.

## References

- K. Gregor and Y. LeCun. *Learning Fast Approximations of Sparse Coding.* ICML, 2010. (LISTA)
- A. Beck and M. Teboulle. *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.* SIAM J. Imaging Sciences, 2009. (ISTA / FISTA)
- R. Tibshirani. *Regression Shrinkage and Selection via the Lasso.* JRSS-B, 1996.
- Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP, 2004. (SSIM)
- M. Aubreville et al. *Mitosis Domain Generalization in Histopathology Images — The MIDOG Challenge.* Medical Image Analysis, 2023. (MIDOG)

## Citation

If you use or refer to this work:

```bibtex
@thesis{pardo2026lista,
  author  = {Pardo García, Enrique},
  title   = {Representación y análisis de células mitóticas en histología H\&E
             mediante autoencoders dispersos basados en LISTA},
  school  = {Universidade da Coruña},
  year    = {2026},
  type    = {Bachelor's Thesis},
  note    = {Supervisors: José Rouco Maseda and Jorge Novo Buján}
}
```

## Acknowledgements and contact

Thanks to the supervisors, José Rouco Maseda and Jorge Novo Buján, and to the **VARPA** research group at the Universidade da Coruña.

**Author:** Enrique Pardo García — *add contact / repository link here.*

**License:** *add a license (e.g. MIT for the code, CC BY 4.0 for the documentation).* The MIDOG data is distributed under its own terms; please consult the dataset's license before redistribution.
