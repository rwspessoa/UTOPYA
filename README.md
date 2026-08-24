# UTOPYA
This repository contains the initial implementation of the algorithms associated with the preprint “UTOPYA: A Multimodal Deep Learning Framework for Physics-Informed Anomaly Detection and Time-Series Prediction” (arXiv:2605.18188), available at [arxiv.org/abs/2605.18188](https://arxiv.org/abs/2605.18188).

![UTOPYA architecture](img/Figure_01.jpg)

## Authors

- Robson Wilson Silva Pessoa — [ORCID: 0000-0003-1603-1453](https://orcid.org/0000-0003-1603-1453) (Department of Chemical Engineering, Norwegian University of Science and Technology, Trondheim 793101, Norway)
- Idelfonso B. R. Nogueira — [ORCID: 0000-0002-0963-6449](https://orcid.org/0000-0002-0963-6449) (Department of Chemical Engineering, Norwegian University of Science and Technology, Trondheim 793101, Norway)

## Funding

This work was funded by the European Union's H2020 Marie Skłodowska-Curie Actions.

| Funder(s) | Award Id(s) |
| --- | --- |
| H2020 Marie Skłodowska-Curie Actions | 101119358 |
| H2020 Marie Skłodowska-Curie Actions | HORIZONMSCA-2022-DN-01 |

## Structure

```
config.py              single source of truth for all paths (see below)
src/
  ablations.py          modality-ablation configs (A1-A12)
  run.py                train + evaluate one config end-to-end (entrypoint)
  run_ablation.py       train + evaluate a batch of configs (A1-A12 by default)
  data/
    loader.py             raw per-experiment time-series loading
    dataset.py             sliding-window Dataset, per-experiment normalisation
    splits.py               leak-free train/val/test split search
    augment.py              training-time jitter/scaling/time-warp
    tabular.py               static operating-point feature extraction
    molecular.py             SMILES -> graph + GC composition cache
    audio.py                 log-mel spectrogram cache
    text_embed.py             SBERT operator-log embedding cache
    nmr.py                    NMR composition feature cache
    image_cache.py            frozen ResNet-18 camera-frame feature cache
  models/
    tcn.py                    TCN encoder
    encoders.py                per-modality encoders + FiLM conditioning
    fusion.py                   cross-modal attention, gated fusion, output heads
    utopya.py                    full UTOPYAModel (wires everything together)
    pretrain.py                  self-supervised TCN pretraining objectives
  training/
    loss.py                      multi-task + physics-informed loss
    train.py                      training loop (curriculum, freeze schedule, dropout)
  evaluation/
    metrics.py                    AUROC/AUPRC/multi-signal fusion
    baselines.py                   PCA / IsoForest / FF-AE / LSTM-AE baselines
    evaluate.py                     evaluation loops
scripts/
  common.py                       shared model+data bootstrap for the scripts below
  train_frozen_extension.py        NMR/image frozen-backbone extension (A14/A15)
  train_reconstruction_head.py     standalone reconstruction-head experiment
  train_phase_head.py              post-hoc phase-classification probe head
  collect_predictions.py           dump per-window predictions/embeddings + fit baselines
  evaluate_all_checkpoints.py      val/test AUROC for every checkpoint under CHECKPOINTS_ROOT
tests/
  smoke_test.py                    data pipeline + TCN encoder + pretrainer
  smoke_test_eval.py                metrics + baselines on a small slice of the data
  smoke_test_full.py                full model forward/backward pass
```

## Data (Zenodo)

This code is validated on the batch-distillation anomaly-detection dataset
by Arweiler et al.:

**Arweiler, J., Jungjohann, I., Muraleedharan, A., Leitte, H., Burger, J.,
Münnemann, K., Jirasek, F., & Hasse, H.** *Batch Distillation Data for
Developing Machine Learning Anomaly Detection Methods* (Version 1.0.2)
[Data set]. Zenodo (2026). https://doi.org/10.5281/zenodo.17395543

The dataset itself (~86 GB) is **not** included in this repository — download
it from the Zenodo record above and point `UTOPYA_DATA_ROOT` at the folder
containing the numbered modality directories (`00_..._Timeseries_Label_...`
through `12_..._Image`), exactly as distributed.

## Configuration

Everything reads paths from `config.py`, which defaults to folders inside
this repository (`./data`, `./checkpoints`, `./outputs`) but can be
overridden without editing any code:

```bash
export UTOPYA_DATA_ROOT=/path/to/arweiler_dataset
export UTOPYA_CHECKPOINTS_ROOT=/path/to/checkpoints
export UTOPYA_OUTPUTS_ROOT=/path/to/outputs
export UTOPYA_DEVICE=cuda   # or cpu
```

## Quickstart

```bash
pip install -r requirements.txt

# Sanity-check that the Zenodo dataset loads correctly end-to-end
python -m tests.smoke_test
python -m tests.smoke_test_eval
python -m tests.smoke_test_full

# Train + evaluate the full multimodal model (A7)
python src/run.py --ablation A7

# Train + evaluate all 12 modality-ablation configs
python src/run_ablation.py

# Extend a trained A7 with the NMR or image modality (frozen backbone)
python scripts/train_frozen_extension.py --modality nmr
python scripts/train_frozen_extension.py --modality image

# Post-hoc probe heads / evaluation utilities (require an A7 checkpoint
# already present under CHECKPOINTS_ROOT/A7 or CHECKPOINTS_ROOT/A7_v2)
python scripts/train_phase_head.py
python scripts/train_reconstruction_head.py
python scripts/collect_predictions.py
python scripts/evaluate_all_checkpoints.py
```

`src/run.py --help` lists every training-procedure flag (augmentation,
focal loss vs. plain weighted cross-entropy, Kendall uncertainty weighting,
TCN freeze schedule, per-modality stochastic dropout rate, physics-loss
weights, curriculum on/off, etc.) — each ablation/variant is a flag, not a
code fork.

## Contact

Robson Wilson Silva Pessoa
Department of Chemical Engineering, Faculty of Natural Sciences,
Norwegian University of Science and Technology (NTNU), Trondheim, Norway.

Supervisor: Dr. Idelfonso Bessa dos Reis Nogueira

## License

This work is dedicated to the public domain under the CC0 1.0 Universal
(CC0 1.0) Public Domain Dedication. See [LICENSE](LICENSE) for the full
legal text.
