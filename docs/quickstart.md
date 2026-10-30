# Basic usage

This guide covers installation, input construction, continuous inference, episode resets and model weights. For a Chinese walkthrough, see [中文使用指南](usage_zh.md). FACA uses a five-value task descriptor and full-state execution history.

## 1. Install from the repository root

Use Python 3.10 or newer, with a PyTorch build appropriate for your device. The checked CPU environment uses Python 3.11 and PyTorch 2.7.0.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/inference.py
```

The command runs an offline demonstration with random weights and synthetic observations; it requires no Isaac Sim, task assets or robot. Typical shapes for the default M3 model are:

```text
variant=m3 device=cpu mode=prior weights=random inputs=synthetic
step=0 latent=(2, 384) action=(2, 3) reset_rows=[]
  alignment=(2, 2) regime_logits=(2, 3)
...
step=2 latent=(2, 384) action=(2, 3) reset_rows=[0]
```

The third measurement starts a new episode in row 0 to demonstrate partial reset. The inputs remain placeholders; this does not simulate assembly performance.

```bash
python examples/inference.py --variant m0 --batch-size 1 --steps 2
python examples/inference.py --variant m1 --seed 7
python examples/inference.py --mode candidates --candidates 8
python examples/inference.py --device cuda:0
python -m faca.parameter_audit
```

The CUDA command requires a working CUDA PyTorch installation. The candidate example scores eight uniformly sampled action sequences and selects the first action from the highest-scoring sequence. It demonstrates candidate scoring; iterative MPPI proposal refinement is outside the example.

## 2. Construct the model and task descriptor

```python
import torch
from faca import FACAConfig, FACA, InferenceSession, task_descriptor, normalize_wrench

device = torch.device('cpu')
model = FACA(FACAConfig(variant='m3')).to(device).eval()
session = InferenceSession(model, batch_size=1)
p = task_descriptor(torch.tensor([0], device=device), 7.986, 0.1, 28.616)
```

`p` has shape `[1,5]`. The inputs are mating type (0: peg-in-hole, 1: hole-onto-peg), held-part transverse width in **mm**, configured radial clearance in **mm**, and nominal held-part axial extent in **mm**. Fixed reference scales produce `[type, log(D/7.986), c/3.887, 2c/D, d/28.616]`. These values stay fixed within an episode. They are not measured fixture offsets or actual surface-gap estimates.

Move the model to its device **before** constructing a session. Use float32 inputs on that device. Scalar descriptor inputs produce shape `[5]`; batched inference requires an explicitly batched mating-type tensor or equivalent broadcasted inputs.

## 3. Supply synchronized state and wrench

```python
state = torch.zeros(1, 17, device=device)
state[:, 3] = 1.0  # identity quaternion: wxyz
raw_tared_tcp_wrench = torch.zeros(1, 6, device=device)
wrench = normalize_wrench(raw_tared_tcp_wrench)
measured = session.observe(state, wrench, p)
```

Replace the synthetic values with the integration's measurements. State contains relative TCP position, quaternion, linear/angular velocity, gripper width and known nominal target position. State positions and descriptor geometry have different units: use the state convention in [the API guide](model_api.md), rather than passing descriptor millimetres into position channels.

The six wrench entries are `[Fx,Fy,Fz,Tx,Ty,Tz]` in N / N m, already tared and expressed at the TCP. `normalize_wrench` divides force by 50 and torque by 5; it does not transform frames, subtract bias, filter or clip. Do not normalize a second time if your environment already returns normalized wrench. State and wrench must refer to the same control sample.

## 4. Predict, choose an action, and advance recurrence

```python
with torch.no_grad():
    action, _ = model.policy(measured.latent, measured.task_context, deterministic=True)
    predicted_next = model.next(measured.latent, action, measured.task_context)
    reward = model.decode(model.reward_logits(measured.latent, action, measured.task_context))
    alignment, force_logits = model.auxiliary(measured)  # M2/M3 only

session.commit_action(action)
# After execution and the next measurement:
next_measured = session.observe(state, wrench, p)
```

The repeated `state` and `wrench` above are placeholders. A real integration reads the response to the executed action before calling `observe` again. The prior action is bounded to `[-1,1]` in each dimension. A controller must map those normalized values to its Cartesian displacement convention; the model does not define a robot-specific millimetre scale. If the controller changes a proposal, commit the **executed** action expressed in model-action coordinates.

Every `observe` must be followed by one `commit_action` before the next observation. The session stores the measured latent and preceding executed action. Do not write `predicted_next` or the end of a candidate rollout into the session's measured recurrence.

The auxiliary alignment is a two-coordinate prediction in metres; force outputs are three logits, optionally converted with `force_logits.softmax(-1)`. These heads do not determine the base controller's action score. They are not calibrated physical state or jam estimates. See [the method guide](method.md).

## 5. Reset episodes

After committing the previous step's action, pass the next episode's first observation with a reset mask:

```python
session.commit_action(torch.zeros(1, 3, device=device))  # completes the preceding observe
new_p = task_descriptor(torch.tensor([1], device=device), 7.986, 0.1, 28.616)
first = session.observe(state, wrench, new_p,
                        reset_mask=torch.tensor([True], device=device))
```

This continues the preceding code block, which ended with `next_measured = session.observe(...)`. Reset clears the row's history, previous latent and previous action before inserting its new observation. Batched masks clear only selected rows. A descriptor may change only on a reset row. Construct a fresh session to clear all state when switching model weights, device or batch size.

## 6. Save and load matching weights

Save the model configuration together with its state dictionary so that weights can be loaded into the matching architecture:

```python
from dataclasses import asdict

torch.save({'config': asdict(model.config), 'state_dict': model.state_dict()}, 'model.pt')
payload = torch.load('model.pt', map_location=device, weights_only=True)
restored = FACA(FACAConfig(**payload['config'])).to(device)
restored.load_state_dict(payload['state_dict'], strict=True)
restored.eval()
restored_session = InferenceSession(restored, batch_size=1)
```

The same save/load format works for random initialization and learned weights. Preserve the observation frames, normalization and action scale associated with a checkpoint. Legacy `WorldModel` weights use different input and head definitions; see [version compatibility](architecture.md#version-compatibility).

## Where to go next

- [Model method](method.md): why the two pathways exist and how prediction differs from a measured update.
- [API reference](model_api.md): tensor shapes, state channels and return values.
- [Parameter audit](parameter_counts.md): module counts and legacy configurations.
- [Source provenance](../PROVENANCE.md): development history and upstream attribution.
