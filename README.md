# FACA: Force-Aware Contact Adaptation

FACA is a task-conditioned, contact-aware world model for robotic assembly. It combines nominal task geometry with recent force–proprioceptive interaction history, updating a recurrent latent state for prediction and action scoring.

This repository contains FACA's neural models, input utilities and inference interfaces. Start with `FACA`, `FACAConfig` and `InferenceSession`.

## Project webpage

The [project page](https://hallyx.github.io/FACA-model/) presents the method, structure diagram and simulation videos.

## Getting started

Install PyTorch for your device and use Python 3.10 or newer. The CPU environment used for checks is Python 3.11, PyTorch 2.7.0, TensorDict 0.11.0 and OmegaConf 2.3.

```bash
python -m pip install -e '.[test]'
python examples/inference.py
python -m pytest -q
```

The example runs synthetic observations through a randomly initialized model. Default output shapes are latent `[2,384]`, action `[2,3]`, alignment `[2,2]` and force-regime logits `[2,3]`.

```python
import torch
from faca import FACA, InferenceSession, task_descriptor, normalize_wrench

model = FACA().eval()
session = InferenceSession(model, batch_size=2)
p = task_descriptor(torch.tensor([0, 1]), 7.986, 0.1, 28.616)
state = torch.zeros(2, 17)
state[:, 3] = 1.0  # identity quaternion, wxyz
wrench = normalize_wrench(torch.zeros(2, 6))
measured = session.observe(state, wrench, p)

with torch.no_grad():
    action, _ = model.policy(measured.latent, measured.task_context, deterministic=True)
    alignment, regime_logits = model.auxiliary(measured)
    successors, score = model.rollout(
        measured.latent, action[None].expand(3, -1, -1), measured.task_context)

session.commit_action(action)
```

Actions are normalized to `[-1,1]`. The controller integration supplies the Cartesian displacement scale and commits the action actually executed. See [Basic usage](docs/quickstart.md) or [中文使用指南](docs/usage_zh.md) for continuous execution, reset handling, candidate scoring and loading matching weights.

## Architecture

[![FACA architecture: task context, contact-aware world model and latent-space MPC](docs/assets/faca_architecture.png)](docs/1.pdf)

A fixed task descriptor supplies nominal geometry context. Four steps of state, wrist wrench and preceding executed actions supply contact evidence. The recurrent state update fuses both pathways into a latent that supports prediction without future force measurements. Auxiliary alignment and force-regime heads read contact context before latent fusion.

| Module | Default dimensions |
|---|---|
| Task encoder | Five-value descriptor → type/metric branches → 64 |
| Execution history | 104 → 128 → 128 → 64 |
| Contact encoder | 468 → 128 → 64 |
| Recurrent state fusion | 532 → 128 → 384 |
| Dynamics | 451 → 256 → 256 → 384 |
| Reward / each online critic | 451 → 256 → 256 → 101 |
| Policy prior | 448 → 256 → 256 → 6 |
| Alignment / force-regime head | 64 → 128 → 2 / 3 |

The system diagram includes MPC and optional correction-assisted calibration. This release provides the neural components, measured-state updates and candidate scoring primitives; controller integration supplies proposal refinement and hardware execution.

## Model variants

| Variant | State update |
|---|---|
| M0 | Current state and static task context |
| M1 | Direct fusion of execution history and recurrent inputs |
| M2 | Separate task/contact pathways and auxiliary heads, with auxiliary loss weights zero during learning |
| M3 | Same inference structure as M2, with contact supervision during learning |

`FACAConfig()` selects M3. Use `FACAConfig(variant='m0')` etc. for other variants. Trainable parameter counts are 1,368,682 (M0), 1,465,130 (M1) and 1,550,703 (M2/M3).

```bash
python examples/inference.py --variant m0 --batch-size 1
python examples/inference.py --mode candidates --candidates 8
python examples/inference.py --device cuda:0
python -m faca.parameter_audit
```

The CUDA option requires a working CUDA PyTorch installation. The candidate example scores uniformly sampled three-step sequences; the full iterative MPPI controller is outside this package.

## Documentation

| Guide | Contents |
|---|---|
| [Basic usage](docs/quickstart.md) / [中文使用指南](docs/usage_zh.md) | Installation, inputs, execution loop, resets and weights |
| [Method](docs/method.md) | Task/contact pathways and latent prediction |
| [Model API](docs/model_api.md) | Tensor layouts, units and output semantics |
| [Architecture](docs/architecture.md) | Network definitions and version compatibility |
| [Parameter counts](docs/parameter_counts.md) | Per-module arithmetic and current/legacy model counts |

## Evaluation and release scope

Our held-out-size evaluation covers ten assembly families with 50 seen and 50 held-out conditions. The evaluated variants achieved success rates of 66.0% (M0), 69.0% (M1), 71.8% (M2) and 76.1% (M3). These results correspond to the research evaluation and its checkpoints. Pretrained weights, benchmark assets and training pipelines are not bundled; runnable examples demonstrate the inference interfaces with random weights.

The legacy `WorldModel` interface remains available for existing configurations. Its input and head definitions differ from `FACA`; [version compatibility](docs/architecture.md#version-compatibility) describes the boundaries. Previous import names remain available as aliases.

## License

FACA builds on Newt/TD-MPC2 and preserves the upstream [MIT license](LICENSE). [Development lineage](PROVENANCE.md) records source history and component attribution.
