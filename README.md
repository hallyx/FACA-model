# FACA model

A model-only extraction of the FACA world model and its task, contact and execution-history encoders. It is derived from FACA archive commit `1574417ec1c352c87e3e1af80edae9b7b6ad235d`; see [PROVENANCE.md](PROVENANCE.md). This repository contains no training pipeline, replay buffer, simulator, planner, trained weights, or experiment data.

## Install and run

Python 3.10+ and a compatible PyTorch installation are required. The CPU smoke tests were run with Python 3.11, PyTorch 2.7.0 and TensorDict 0.11.0. Install PyTorch for your platform, then:

```bash
pip install -e '.[test]'
pytest -q
```

The following uses randomly initialized weights and a small demonstration configuration; its outputs are not a usable trained robot policy.

```python
import torch
from tensordict import TensorDict
from faca import WorldModel
from faca.example_config import example_config

model = WorldModel(example_config('m1')).eval()
obs = TensorDict({'state': torch.zeros(2, 17), 'wrist_wrench': torch.zeros(2, 6)}, batch_size=[2])
task = torch.zeros(2, 6)
history = {'state_history': torch.zeros(2, 4, 14),
           'force_history': torch.zeros(2, 4, 6),
           'action_history': torch.zeros(2, 4, 3)}
with torch.no_grad():
    z = model.encode(obs, task, history, torch.zeros(2, 16), torch.zeros(2, 3))
    action, _ = model.pi(z, task)
```

## Observation contract

The example uses the archived `state_goal_wrench_history_v1` contract: `state` has 14 canonical TCP-state values followed by the 3D goal TCP position in the socket frame. `wrist_wrench` is a separate normalized 6D sensor channel; M1–M3 receive it through the explicitly supplied four-step history, not directly from the current sensor field. The caller must construct correctly aligned histories and previous latent/action inputs. M0 does not use wrench history. This interface differs from the 20D single-frame V3 contract in the main FACA project; checkpoints and observations must not be relabeled or mixed between contracts.

M2/M3 additionally expose alignment and three-class contact heads. `pi` is a model policy head, not the original TD-MPC2 planning controller. Supplying trained weights, normalization, environment integration and any planning logic is the caller's responsibility. Do not load untrusted PyTorch checkpoints.

## Release status

This is a model extraction prepared locally for review, not a published release or a reproduction of training results. Public release still requires the project owner to confirm rights for all contributions and dependencies, select a public hosting location and review any artifacts to be distributed. See [PROVENANCE.md](PROVENANCE.md) and [LICENSE](LICENSE).
