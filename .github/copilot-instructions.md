# Copilot instructions for I-JEPA (ijepa)

Purpose
- Help Copilot/AI agents be immediately productive in this repo by summarizing the architecture, key workflows, project-specific patterns, and concrete examples.

Quick start (examples)
- Single-GPU (local):
  - `python main.py --fname configs/in1k_vith14_ep300.yaml --devices cuda:0`
- Multi-GPU local (example 3 GPUs):
  - `python main.py --fname configs/in1k_vith14_ep300.yaml --devices cuda:0 cuda:1 cuda:2`
- SLURM / cluster (submitit example):
  - `python main_distributed.py --fname configs/in1k_vith14_ep300.yaml --folder $LOG_DIR --partition $PART --nodes 2 --tasks-per-node 8 --time 1000`

Where to change experiments
- All experiments are config-driven using YAML files in `configs/`.
  - Important sections: `meta`, `data`, `mask`, `optimization`, `logging`.
  - Always set `data.root_path` and `logging.folder` before running.
  - The code dumps the resolved params to `params-ijepa.yaml` in the experiment folder when training starts.

Checkpoints & logs
- Checkpoints written into `logging.folder` with names:
  - `{{write_tag}}-latest.pth.tar` and `{{write_tag}}-ep{epoch}.pth.tar`
- Training logs are CSVs (`{tag}_r{rank}.csv`) — see `src/utils/logging.py` for format.

Key files & components (read these first)
- Entrypoints: `main.py` (local), `main_distributed.py` (submitit/SLURM)
- Training loop: `src/train.py` (main logic, DDP usage, logging)
- Model & initialization: `src/helper.py` (init_model, init_opt, load_checkpoint)
- Model definitions: `src/models/vision_transformer.py`
- Masking utilities: `src/masks/` (e.g., `multiblock.py`, `utils.py`) — masking is central to IJEPA.
- Data loader: `src/datasets/imagenet1k.py` (expects `root_path` + `image_folder` layout)
- Distributed helpers: `src/utils/distributed.py`
- Config templates: `configs/*.yaml` (use these as authoritative examples)

Project-specific conventions & patterns
- Config-first design: prefer adding config keys over adding ad-hoc CLI args.
- Deterministic seeds: global seed set at top of `src/train.py` — check and modify there when changing reproducibility.
- DDP usage: code expects one visible GPU per process (handles `SLURM_LOCALID` and `CUDA_VISIBLE_DEVICES` accordingly).
- Mixed precision: controlled via `meta.use_bfloat16` in configs (bfloat16 supported by GradScaler when enabled).
- Weight init: models use `trunc_normal_` pattern in `src/helper.py`.
- Checkpoint loading: `helper.load_checkpoint` handles encoder/predictor/optimizer/scaler; follow its structure for compatibility.

Developer workflow & testing
- Requirements: Python 3.8+, PyTorch 2.x, torchvision, submitit (see `README.md`).
- No dedicated unit test suite in repo — add tests under a `tests/` folder if you add behavior that needs automated coverage (see `CONTRIBUTING.md`).
- For quick local iteration, create a tiny YAML config (small batch/epochs) and run `main.py` on a single GPU.

PR guidance for agents (concrete)
- If changing hyperparameters or adding experiments, update or add a `configs/*.yaml` and the README example if user-facing.
- When touching model shapes/weights, ensure `init_weights` behavior remains consistent and update `load_checkpoint` compatibility notes.
- When adding dataset changes, update dataset loader docs and example `data.root_path` usage.
- Always run a short smoke training (1–2 iterations) locally and confirm checkpoint writes and CSV logs.

How to ask this repo's maintainers (good prompts for Copilot/agent)
- "Add a new CLI flag `--seed` that sets numpy and torch seeds and dump it into `params-ijepa.yaml`. Modify `main.py` and `src/train.py` to read it from the YAML if present."
- "When resuming from checkpoint, print the loaded epoch and optimizer lr schedule steps — update `helper.load_checkpoint` and add tests or a short smoke run." 

Safety & performance notes
- Large configs expect many GPUs (see top of `README.md`). For changes that might break large-scale runs, test with scaled-down configs first.
- Be cautious with external data paths and avoid changing data paths without documenting expected layout (ImageNet folder names are used directly).

References
- `README.md` (project overview + launch examples)
- `CONTRIBUTING.md` (PR + style guidelines: 4-space indentation, 80-char line length, PEP8)

If anything here is unclear or you'd like more examples (e.g., a short smoke `configs/*.yaml` for CI), tell me which area to expand.  

---
*Generated for Copilot/AI agents: use `#codebase` search and check the files above before making changes.*
