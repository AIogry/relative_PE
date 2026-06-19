# HIPE

Code release for the ICML 2026 paper:

**The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency**

This repository provides the language-modeling and downstream-adaptation code used for the paper's main open-source path. The release is centered on:

- HIPE with fixed and learnable sigma
- restricted-budget language modeling on WikiText-103 and C4
- larger-scale 300M C4 pretraining
- low-resource SST-2 LoRA adaptation
- compatibility with local/global attention ablations

The synthetic mechanism experiments from `Exp 1` are kept in the repo history, but they are not the focus of this open-source release.

## Main capabilities

- `train_exp2_wikifull.py`: WikiText-103 language modeling with RoPE, HIPE, XPos, YaRN, and optional local attention.
- `train_exp2_c4full.py`: C4 language modeling with the same interface, plus learnable-sigma HIPE support.
- `finetune_sst2.py`: SST-2 LoRA fine-tuning for RoPE and learnable-sigma HIPE checkpoints.
- `download_wiki_data.py`, `download_sst2.py`: portable dataset preparation helpers.
- `scripts/release/`: clean shell entrypoints for public reproduction.
- `OLMo/`: the local model implementation, including the unified HIPE, YaRN, and local/global attention logic.

## Setup

```bash
pip install -r requirements.txt
```

For the fuller author environment, see `2026_1_19_new_requirements.txt`.

## Data and outputs

The release scripts use:

```bash
export PE_DATA_DIR=/path/to/data
export PE_ARTIFACTS_DIR=/path/to/artifacts
```

Defaults:

- `PE_DATA_DIR=./data`
- `PE_ARTIFACTS_DIR=./artifacts`

## Recommended release path

### 1. Prepare datasets

WikiText-103:

```bash
bash scripts/release/prepare_wikitext.sh
```

SST-2:

```bash
bash scripts/release/prepare_sst2.sh
```

C4:

Place your processed C4 subsets at:

```text
${PE_DATA_DIR}/c4/c4_30M_train
${PE_DATA_DIR}/c4/c4_30M_validation
```

### 2. Reproduce Exp 2: restricted-budget language modeling

WikiText-103:

```bash
MODEL_SIZE=60M SEQ_LEN=1024 \
EXTRA_ARGS="--use_scaled_rope --sigma 500.0 --rope_scaling_threshold 3" \
bash scripts/release/run_exp2_wikitext.sh
```

C4:

```bash
MODEL_SIZE=60M SEQ_LEN=1024 \
EXTRA_ARGS="--use_scaled_rope --sigma 500.0 --rope_scaling_threshold 3" \
bash scripts/release/run_exp2_c4_restricted.sh
```

For baselines, pass flags such as `--xpos`, `--yarn`, `--alibi`, or no extra HIPE arguments for RoPE.

### 3. Reproduce Exp 3: larger-scale learnable-sigma HIPE

300M C4 pretraining with learnable sigma:

```bash
MODEL_SIZE=300M SEQ_LEN=512 SIGMA=200 THRESHOLD=7 \
bash scripts/release/run_exp2_c4_learnable_sigma.sh
```

You can also run the fixed-sigma larger-scale C4 setting with:

```bash
MODEL_SIZE=300M SEQ_LEN=2048 \
EXTRA_ARGS="--use_scaled_rope --sigma 700 --rope_scaling_threshold 7" \
bash scripts/release/run_exp3_c4.sh
```

### 4. Reproduce SST-2 LoRA adaptation

First point `BASE_MODEL_PATH` to a pretrained checkpoint. For example, a learnable-sigma 300M C4 checkpoint:

```bash
BASE_MODEL_PATH=/path/to/model.pt \
MODEL_VARIANT=hipe \
SEQ_LEN=512 \
FEW_SHOT=100 \
SIGMA=200 \
THRESHOLD=7 \
bash scripts/release/run_exp3_sst2_lora.sh
```

For a RoPE baseline:

```bash
BASE_MODEL_PATH=/path/to/model.pt \
MODEL_VARIANT=rope \
SEQ_LEN=512 \
FEW_SHOT=100 \
bash scripts/release/run_exp3_sst2_lora.sh
```

## Local/global attention ablations

The unified `OLMo/olmo/` implementation supports local/global attention directly through config flags exposed in the training scripts:

```bash
EXTRA_ARGS="--local_window_size 256 --num_local_layers 4 --use_scaled_rope --sigma 100 --rope_scaling_threshold 3" \
bash scripts/release/run_exp2_c4_restricted.sh
```

This is the recommended path for reproducing the local/global attention compatibility experiments without switching branches.

## Notes

- The open-source release is intentionally organized around the paper's language-modeling and adaptation results.
- `learnable_sigma`, `YaRN` compatibility, and local/global attention are unified into the same `OLMo/olmo/` implementation on this branch.
- Legacy SLURM launchers and older exploratory scripts are still present for reference, but `scripts/release/` should be preferred.

## Citation

If you use this repository, please cite the ICML 2026 paper.

```bibtex
@inproceedings{zhou2026hippocampal,
  title     = {The Hippocampal Place Field Gradient: A Bio-inspired Framework Building Multiscale Representation for Better Sample Efficiency},
  author    = {Shujun Zhou and Junrong Qi and Guozhang Chen},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {2026}
}