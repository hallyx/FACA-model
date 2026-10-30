# Model parameter counts

Count model parameters and verify the module arithmetic:

```bash
python -m faca.parameter_audit
```

The command builds each variant, counts unique parameters with `requires_grad=True`, and checks every module against independent layer arithmetic. It also counts the legacy configurations. No checkpoint or simulator is required.

## Result

Current and legacy FACA models have different task descriptors, state histories and auxiliary-head inputs. Their totals are recorded separately below; all values use the same trainable-parameter counting rule.

| Variant | FACA | Legacy WorldModel | Legacy expected total | Legacy minus current |
|---|---:|---:|---:|---:|
| M0 | 1,368,682 | 1,378,256 | 1,378,256 | 9,574 |
| M1 | 1,465,130 | 1,473,168 | 1,473,168 | 8,038 |
| M2/M3 | 1,550,703 | 1,656,661 | 1,656,661 | 105,958 |

M0/M1 state encoders use one 128-wide hidden block. Their state-encoder width is independent of the two 256-wide prediction-head blocks. This baseline setting is recorded in the resolved experiment configurations and model architecture output.

## Layer accounting

For a biased Linear `a → b`, the count is `(a + 1) b`. Affine LayerNorm adds `2b`. Mish, SimNorm and distribution-decoding support buffers add no trainable parameters. The task embedding contributes `2 × 8 = 16`. Only online critics are trainable in the historical model; frozen target critics are excluded by the stated counting rule.

| Module | Current M3 | Historical M3 | Explanation |
|---|---:|---:|---|
| Task encoder | 8,400 | 18,358 | Five-value descriptor and 40-wide fusion versus six-value descriptor, yaw branch, 48-wide fusion and registered residual/decoder modules |
| History encoder | 38,720 | 37,184 | Four 17D states versus four 14D states; `4 × 3 × 128 = 1,536` additional weights |
| Contact encoder | 68,544 | 68,160 | 468 versus 465 inputs; `3 × 128 = 384` additional weights |
| State fusion | 118,016 | 117,632 | 532 versus 529 inputs; `3 × 128 = 384` additional weights |
| Alignment head | 8,834 | 57,986 | 64D contact context versus 448D latent+task; `(448 − 64) × 128 = 49,152` fewer weights |
| Force-regime head | 8,963 | 58,115 | Same 49,152-weight reduction |
| Dynamics | 281,984 | 281,984 | Same 451→256→256→384 backbone, including inherited output LayerNorm |
| Reward | 208,485 | 208,485 | Same 451→256→256→101 network |
| Three online critics | 625,455 | 625,455 | Three reward-shaped networks |
| Policy prior | 183,302 | 183,302 | Same 448→256→256→6 network |
| **Total** | **1,550,703** | **1,656,661** | Every parameter assigned to a module |

The task encoder's historical residual and decoder remain registered with `requires_grad=True` even when their optional objectives/routes are disabled, so they are included in the historical total.

For M0, historical state encoding has 60,672 parameters, versus 61,056 with the three added goal coordinates. For M1 the corresponding values are 118,400 and 118,784. All other differences are shared with M3 where applicable. Thus the differences are exactly:

```text
M0:    (18,358 − 8,400) − 384                         =   9,574
M1:    (18,358 − 8,400) − 384 − 1,536                 =   8,038
M2/M3: (18,358 − 8,400) + 2×49,152 − 2×384 − 1,536   = 105,958
```

## Legacy configurations and development records

The FACA development repository records model architectures and configurations in:

- `reports/force_comparison_budget_20261010/m0_84203.log`: M0 state width 14, task encoder 18,358, encoder 60,672, learnable total 1,378,256.
- `logs/wrist_wrench_v3/tasksize10_v2_1_m0_m3_mt_true_seed1_matched_20260920_1710/training_m1.log`: learnable total 1,473,168.
- `reports/force_comparison_budget_20261010/m3_84203.log`: history input 92, contact input 465, state-fusion input 529, auxiliary input 448, learnable total 1,656,661.
- `docs/paper_evidence_20261008/resolved_configs.csv`: state encoder hidden width 128 with `num_enc_layers=2` (one hidden block and one output), prediction width 256, latent width 384 and three critics.

These development records remain outside the model release. `historical_count_config()` selects the corresponding `WorldModel` settings: legacy observation width 14 for M0 or 20 for M1–M3, history-state width 14, task vector width 6 and latent+task auxiliary heads.

## Interpretation

The current model uses the five-value descriptor, full 17D state history and contact-context auxiliary heads. The legacy profile uses a six-value descriptor, 14D state history and latent-plus-task auxiliary heads. The table and layer arithmetic above keep these configurations explicit rather than treating a matching total as model compatibility.

The project research PDF records the legacy totals alongside the five-value-interface architecture description. This is a version mismatch in that record: the current release totals are the FACA column above. Parameter-count checks validate network structure, while benchmark performance requires the matching weights and evaluation configuration.
