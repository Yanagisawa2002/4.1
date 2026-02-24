SPEC.md

Project: HO-regularized R-GCN for Drug–Disease Indication Prediction

1. Objective

We train a heterogenous graph neural network (R-GCN) to perform binary classification of drug–disease indication edges.

We incorporate High-Order (HO) mechanistic quadruplets as auxiliary structural supervision.

HO MUST act only as representation-level regularization.
HO MUST NOT modify pair labels.
HO MUST NOT be used at inference.

2. Data Definition
2.1 KG

Nodes:

drug

disease

protein

pathway

Edges:

multiple biological relations

target relation: indication(drug, disease)

2.2 HO (A-class only)

Each HO quadruplet:
(d, protein, pathway, dis)

Constraint:
(drug=d, disease=dis) MUST be a positive indication edge in KG.

Non-HO paths MUST NOT be treated as negative samples.

3. Split Rules (CRITICAL)

Step 1:
Split KG indication positive edges into:

train

val

test

Split types:

random

cross-drug

cross-disease

Negative sampling:
For each positive edge, sample exactly K=1 negative edge.
Negatives must not overlap with any known positive.

Step 2:
Derive HO splits strictly from KG splits:

HO_train = {q | (d,dis) ∈ KG_pos_train}
HO_val = {q | (d,dis) ∈ KG_pos_val}
HO_test = {q | (d,dis) ∈ KG_pos_test}

Training MUST use only HO_train.

Evaluation MUST NOT use HO_val or HO_test.

Cross-drug constraint:
train drugs ∩ test drugs = ∅

Cross-disease constraint:
train diseases ∩ test diseases = ∅

All splits must be reproducible with a fixed random seed.

4. Model Architecture
4.1 Encoder

Use R-GCN over the heterogeneous KG.

Output:
Embedding h_v for each node v.

4.2 Pair Head

Score:
s(d, dis) = MLP([h_d || h_dis || h_d ⊙ h_dis])

Loss:
L_pair = BCEWithLogits(s, y)

4.3 HO Hyperedge Head

For each HO quadruplet q=(d, pr, pw, dis):

Construct hyperedge representation:

z_q = MLP([h_d || h_pr || h_pw || h_dis])

Projection head:
Proj(h_x)

Component-level InfoNCE:

For each component x ∈ {d, pr, pw, dis}:

Positive:
Proj(h_x)

Negatives:
In-batch nodes of same type

L_HO(q) = Σ_x InfoNCE(z_q, Proj(h_x))

5. Training Objective

Total loss:

L = L_pair + λ · L_HO

Default:
λ = 0.1

L_HO MUST NOT alter pair logits directly.

HO loss gradients should primarily update encoder.

6. Batch Strategy

Each training step MUST include:

Pair mini-batch:
B positive edges + K=1 negatives per positive

HO mini-batch:
M quadruplets sampled from HO_train

Both losses computed in same forward pass.

7. Bias Mitigation

HO sampling MUST be balanced by drug (default) or disease.

HO loss MUST include frequency-based weighting:

w(q) = 1 / sqrt(freq(protein) + freq(pathway))

8. Evaluation

Inference MUST use only pair head.

Report:

AUROC

AUPRC

Run on:

random

cross-drug

cross-disease

Include ablations:

Base (no HO)

+HO

+HO (no debias)

9. Non-Negotiable Constraints

HO is auxiliary supervision only.

No negative HO labels.

No HO leakage across splits.

K = 1 negative sampling.

R-GCN encoder only.

Reproducible splits.

10. Current IO Contract (Temporary, Project-Specific)

IMPORTANT:
At the current stage, we DO NOT use a standardized/canonical IO schema.
All code must follow the project files and columns below.

Current data layout:

`data/KG/nodes.csv`
- columns: `id,type,name,source`
- node ID examples: `drug::DB00001`, `disease::5044`, `gene/protein::1129`, `pathway::R-HSA-390648`

`data/KG/*.csv` edge tables (current KG relation tables)
- primary schema: `relation,x_id,x_type,y_id,y_type`
- examples:
  - `indication_data_subset.csv` (target relation positives)
  - `kg_filtered_subset_ext_drugprotein.csv` (heterogeneous KG edges)

`data/HO/HO.csv`
- current schema includes:
  - `drugbank_id`, `protein_id`, `pathway_id`, `disease_id`
  - plus additional text/metadata columns (e.g., rationale fields)

Current normalization rules in pipeline:
- `drugbank_id` values like `DB00334` must be normalized to `drug::DB00334`
- `pathway_id` values like `http://bioregistry.io/reactome:R-HSA-390648` must be normalized to `pathway::R-HSA-390648`

Split output schema (internal training IO) remains:
- KG pairs: `drug,disease`
- HO quads: `drug,protein,pathway,disease`

Until further notice:
- Do not assume “standard IO” column names unless explicitly mapped.
- Treat this section as authoritative for current project IO.

END OF SPEC
