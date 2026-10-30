# Model inputs and inference API

Start with [Basic usage](quickstart.md) or [中文使用指南](usage_zh.md) for a complete invocation sequence. [The method guide](method.md) explains the task/contact pathways and recurrent prediction workflow.

## Descriptor and observation

`task_descriptor(type, width_mm, clearance_mm, axial_extent_mm)` produces

```text
p = [type, log(D / 7.986), c / 3.887, 2c / D, d / 28.616]
```

All input lengths use millimetres. Type is 0 for peg-in-hole and 1 for hole-onto-peg. `D` is the held-part transverse-width proxy; `c` is a configured radial-clearance descriptor; `d` is nominal held-mesh axial extent, not target insertion depth. Inputs broadcast using PyTorch broadcasting rules. Keep the resulting descriptor fixed within each episode.

The 17D state uses the following channel layout:

| Slice | Quantity |
|---|---|
| 0:3 | TCP position relative to the nominal reference, expressed in a common coordinate basis |
| 3:7 | TCP quaternion, wxyz |
| 7:10 | TCP linear velocity |
| 10:13 | TCP angular velocity |
| 13:14 | Gripper width |
| 14:17 | Known nominal target TCP position |

State position and gripper-width channels use metres; linear and angular velocity use metres/second and radians/second, respectively. Descriptor constructor geometry uses millimetres. Use float32 tensors on the model device and a normalized quaternion in wxyz order.

The runtime wrapper must establish the coordinate frames and episode-fixed reference perturbation. This package does not infer a socket transform from sensor readings. `normalize_wrench` only scales an already tared, TCP-framed six-vector; it does not tare a sensor or transform its frame.

## Measured state update

```python
measured = model.encode(
    state, descriptor, history=history,
    previous_latent=previous_measured_latent,
    previous_action=previous_executed_action,
)
```

`state` is `[B,17]`; `descriptor` is `[B,5]`. The returned `MeasuredState` contains `latent [B,384]`, `task_context [B,64]` and (for M2/M3) `contact_context [B,64]` at default dimensions. Explicit histories are `[B,4,17]` for state, `[B,4,6]` for normalized wrench, and `[B,4,3]` for preceding actions. They are ordered oldest to newest. The current observation occupies the last row, paired with the action that led to it.

At reset, absent history and previous latent/action are zero. `InferenceSession.observe` inserts the reset observation into the last history row, while earlier rows remain zero. At the next step, `commit_action` supplies the action actually executed; it may differ from the policy proposal. Partial resets clear only the specified rows. Episode descriptors may change only for reset rows. Construct a new session if batch size or model device changes; call `model.eval()` for execution.

## Prediction and model outputs

| API | Inputs after measured update | Output |
|---|---|---|
| `next(z, a, c_task)` | Latent, action, static task context | Next latent |
| `reward_logits(z, a, c_task)` | Same | 101 reward logits |
| `value_logits(z, a, c_task)` | Same | `[3,B,101]` online-critic logits |
| `decode(logits)` | Reward or value logits | Scalar per distribution, trailing dimension 1 |
| `policy(z, c_task)` | Latent and task context | Tanh Gaussian sample and corrected log density |
| `policy(..., deterministic=True)` | Same | Tanh mean and `None` log density |
| `auxiliary(measured)` | Measured contact context | Alignment `[B,2]` and force-regime logits `[B,3]` |
| `rollout(z, actions, c_task)` | Actions `[H,B,3]` | Successor latents `[H,B,384]`, score `[B,1]` |

Candidate batch dimensions may replace B in the stateless APIs. Broadcast or expand state and task context consistently before scoring candidates. `rollout` does not change the session's measured recurrence; commit only executed real actions. Its terminal value remains stochastic because it samples a prior action and two critics. Use a fixed PyTorch seed for repeatable demonstrations.

M0/M1 state encoders use a single 128-wide hidden block by default (`baseline_hidden_dim`), independently of the 256-wide prediction heads. M0 has no history or recurrence in its state encoder. M1 concatenates history and recurrence directly. M2/M3 add the separate contact encoder and auxiliary heads; they have identical inference structures, differing in the supervision used to produce their weights. All distributed examples use random weights.
