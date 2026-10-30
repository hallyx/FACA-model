# Model architecture

FACA combines nominal task context and measured interaction evidence in a recurrent world model. `FACAConfig` defines the network dimensions and component variant. `FACA` provides measured-state encoding, latent dynamics, prediction heads and action-sequence scoring.

## Inputs and pathways

| Input | Shape | Purpose |
|---|---|---|
| Nominal task descriptor | `[B,5]` | Mating type and four normalized geometry features |
| Current state | `[B,17]` | TCP state and known nominal target position |
| State history | `[B,4,17]` | Four chronological state samples |
| Wrench history | `[B,4,6]` | Normalized TCP wrist force/torque |
| Preceding-action history | `[B,4,3]` | The executed action that produced each observation |
| Previous measured latent | `[B,384]` | Recurrent execution state |
| Previous executed action | `[B,3]` | Most recent real action |

The task encoder produces a static 64D context. The history encoder produces a 64D feature. The contact encoder combines that feature with current state and previous measured latent/action, producing a 64D contact context. State fusion integrates these inputs into a 384D latent.

The alignment and force-regime heads read the measured contact context before state fusion. Imagined dynamics, reward, value and policy heads use the latent and static task context; they do not require future sensor measurements.

## Default dimensions

| Module | Input width | Hidden widths | Output width |
|---|---:|---|---:|
| Task encoder | 5 | Type embedding 8; metrics 32,32; fusion 64,64 | 64 |
| History encoder | 104 | 128,128 | 64 |
| Contact encoder | 468 | 128 | 64 |
| Recurrent state fusion | 532 | 128 | 384 |
| Dynamics | 451 | 256,256 | 384 |
| Reward | 451 | 256,256 | 101 |
| Each of three critics | 451 | 256,256 | 101 |
| Policy prior | 448 | 256,256 | 6 |
| Alignment | 64 | 128 | 2 |
| Force regime | 64 | 128 | 3 |

Hidden blocks use Linear–LayerNorm–Mish. State and dynamics output blocks use SimNorm groups of eight. The dynamics output layer includes LayerNorm before SimNorm. Reward and value use 101 bins over symlog support [−10,10]; the policy prior uses a tanh-squashed Gaussian with log standard deviation in [−10,2].

M0 encodes current state and task context. M1 adds history and recurrence by direct concatenation. Their state encoders use a single 128-wide hidden block. M2/M3 add the separate contact pathway and auxiliary heads; their inference structures are identical, with learned weights distinguished by auxiliary supervision.

The trainable totals are 1,368,682 (M0), 1,465,130 (M1) and 1,550,703 (M2/M3). [Parameter counts](parameter_counts.md) gives the module arithmetic and legacy-version comparison.

## Online state and outputs

`InferenceSession` manages histories and caches the previous measured latent and executed action. At reset, unavailable histories and recurrent inputs are zero. The first real observation fills the newest history row. Partial resets clear only selected batch rows.

Alignment predicts held-part lateral displacement in the ground-truth fixed-part mating frame, in metres. Force-regime outputs represent free, nominal contact and high force. Their supervision uses plug/socket pair-force norm with a causal three-sample median and thresholds at 1 N and 10 N. The runtime wrist wrench is a separate tared TCP measurement, normalized by 50 N / 5 N m with no temporal filtering or clipping before history insertion.

`rollout` scores supplied action sequences using discounted decoded rewards and terminal value. Default discount is 0.95; the MPC horizon is three steps. Terminal value averages two randomly chosen online critics at a sampled prior action. The full MPC evaluation configuration uses 512 candidates, 24 prior trajectories, six iterations, 64 elites and temperature 0.5. This package provides the model scoring operation; proposal refinement and hardware control belong to the controller integration.

## Version compatibility

`WorldModel` retains the earlier model interface. It has a six-value task descriptor, a default 14D state history, and auxiliary heads that consume latent plus task context. Its state-goal configuration can accept 17D current state while still using 14D history. Its legacy count configuration uses 14D current state, with appended wrench for M1–M3.

These input and head differences affect checkpoint shapes and semantics. Load weights into the matching class/configuration with `strict=True`, and retain the associated observation frames, normalization and action scale. A matching observation width or parameter total alone is insufficient to identify a model version.

The previous names `PaperFACA` and `PaperConfig` remain aliases of `FACA` and `FACAConfig`; `faca.paper` and `examples/paper_inference.py` remain compatibility entry points. They do not define separate networks. New integrations should use `from faca import FACA, FACAConfig` and `examples/inference.py`.

The optional residual adapter remains available as `faca.models.TaskConditionedLatentContactResidualAdapter` for the legacy dynamics interface. It is not attached to `FACA` automatically. Its configuration is distinct from the 389,129-parameter adapter recorded in the target-domain calibration experiments; parameter and checkpoint compatibility must be established before use.

## Verification

Checks cover layer dimensions and counts, history/action alignment, partial reset, auxiliary gradient routing, sensor-free imagined prediction, and weight serialization. These checks validate the model API. Benchmark performance requires the corresponding learned weights and evaluation setup.

For runnable examples, see [Basic usage](quickstart.md) or [中文使用指南](usage_zh.md). For development lineage and upstream attribution, see [PROVENANCE.md](../PROVENANCE.md).
