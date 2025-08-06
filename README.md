# Augmentation lists(Paper with codes)

1. BootGen-ESM
Paper: Kim et al., NeurIPS 2023 (https://github.com/kaist-silab/bootgen)
Core Idea: Train a score-conditioned generator (ESM-1b) to sample realistic sequences conditioned on a target property.
Implementation: We skip training and instead sample mutations directly from ESM-1b logits at masked positions.
Hyper-params: mag = fraction of positions to mask; T = temperature controlling diversity.

2. NaNa-BLOSUM
Paper: Huang et al., GNN augmentation (https://github.com/r08b46009/Code_for_MIGU_NANA/tree/main)
Core Idea: Replace residues with BLOSUM62 “semantic neighbours” to preserve physicochemical context.
Implementation: Per-residue coin-flip → pick neighbour with BLOSUM62 ≥ 1.
Hyper-params: mag = probability of replacement; threshold = 1 (can be tuned).

3. NTA-Codon
Paper: Minot & Reddy, Bioinformatics Advances 2022 (https://github.com/minotm/NTA)
Core Idea: Introduce silent or non-silent mutations at codon level while blocking STOP codons.
Implementation: Back-translate → mutate nucleotides → re-translate → STOP filter.
Hyper-params: mag = nucleotide mutation rate.

4. Spider-Toxin
Paper: Lee et al., IJMS 2021 (https://github.com/bzlee-bio/NT_estimation)
Core Idea: Use domain-specific substitution matrix learned from spider neurotoxins.
Implementation: Sample each residue from the published 20 × 20 spider-toxin matrix.
Hyper-params: mag = substitution probability.

5. RSA-Retrieve
Paper: Chang et al., “Retrieved Sequence Augmentation” (2023) (https://github.com/chang-github-00/RSA)
Core Idea: Retrieve k nearest homologues (ESM-1b space) and splice in a segment.
Implementation: Offline ESM-1b index → k-NN search → copy random segment.
Hyper-params: mag unused; instead k (neighbours) & seg_len (length to splice).

6. PreIS-HA
Paper: CBRC-lab, 2023 (https://github.com/CBRC-lab/PreIS)
Core Idea: Subtype-conditional masked LM – generate HA sequences while preserving influenza subtype label.
Implementation: Use the released PreIS-HA masked LM to sample masked positions.
Hyper-params: mag = fraction of positions to mask.

7. PCV-Rotate
Paper: Kucheryavskiy et al., Analytica Chimica Acta 2023 (https://github.com/svkucheryavski/pcv)
Core Idea: Apply orthogonal Procrustes rotation to k-mer vectors to create orientation variants.
Implementation: Tri-peptide vector → rotation → nearest-AA mapping.
Hyper-params: mag = window-size fraction; k-mer size = 3 (fixed).

8. DRO-Fill
Paper: Moon et al., 2024 (https://github.com/HaeunM/peptide-imputation-inference)
Core Idea: Impute missing residues using neighbour consensus under a doubly-robust framework.
Implementation: Randomly mask residues → fill with most-common neighbour (±radius).
Hyper-params: mag = mask rate; radius = context window.
