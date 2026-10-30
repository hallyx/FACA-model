# Method guide

FACA updates a recurrent world model from nominal task context and measured interaction evidence, then uses latent prediction for action scoring. This guide describes the model inputs, update rules and prediction heads. See [basic usage](quickstart.md) for executable examples and [architecture notes](architecture.md) for compatibility with historical models.

## Nominal task context and measured interaction evidence

A known task specification is useful before contact occurs. The five-value task descriptor supplies mating type, transverse scale, configured clearance and nominal axial extent. The descriptor does not specify the grasp offset, fixture error or friction actually encountered in an episode.

FACA therefore has two directly supplied conditioning pathways:

- The task encoder maps the fixed descriptor to `c_task`, a 64D nominal task context.
- The history and contact encoders summarize recent state/action/wrench responses into `c_contact`, a 64D measured contact context.

This separation is architectural. It does not make the contexts identifiable physical parameters or calibrated estimates of friction, grasp offset or contact mode. Contact context also depends on current state, previous measured latent and preceding action, so it is not a force-only embedding.

## Inputs and measurement timing

The runtime observation comprises a 17D state and a separate 6D wrist wrench, for 23 sensor/state values in total. The state includes TCP pose, velocities, gripper width and the known nominal target position. The wrist wrench is tared and TCP-framed, then scaled by 50 N for force and 5 N m for torque. No clipping or temporal filter is applied before the history buffer.

At control step `t`, the newest history row is `(s_t, w_t, a_{t-1})`: the action in the row is the one that produced the measured response. It is not the new action about to be proposed. Four oldest-first rows are flattened to `4 × (17 + 6 + 3) = 104` inputs.

At an episode boundary, unavailable rows, the previous latent and the previous action are zero. The first real observation occupies the newest row. Subsequent samples shift the buffer forward. `InferenceSession` implements this execution-time bookkeeping. The four-step gradient-free replay burn-in reconstructs training histories. Online inference can propose an action after the first measured update.

## Measured-state update

```text
c_task      = T(p)
h_t         = H(history_t)
c_contact_t = C(s_t, h_t, z_measured_(t-1), a_executed_(t-1))
z_measured_t = E(s_t, z_measured_(t-1), a_executed_(t-1), c_task, c_contact_t)
```

The history encoder is a feed-forward MLP. Recurrence comes from the cached previous measured latent. `FACA.encode` returns `MeasuredState` containing the updated latent and both contexts. The default M3 update uses 468 inputs for contact encoding and 532 for state fusion, producing a 384D latent with SimNorm groups of eight.

`InferenceSession.observe` updates the history and measured recurrence together. `commit_action` records the action actually executed. If a controller clips, rescales or replaces the proposed action, convert the final executed displacement back to the model's action convention before committing it. A batch row may change its descriptor only when that row begins a new episode.

## Imagined prediction

After the newest real measurement establishes `z_measured_t`, the world model can predict a candidate action sequence:

```text
z_imagined_(t+1) = D(z_measured_t, a_t, c_task)
reward_logits_t = R(z_measured_t, a_t, c_task)
prior_action_t  ~ pi(. | z_measured_t, c_task)
value_logits_t  = Q(z_measured_t, a_t, c_task)
```

At subsequent imagined steps, dynamics use the preceding predicted latent. The static task context is available throughout the horizon. Future measured wrench and future contact context are not available and are not required by `next`. The encoded latent carries the current interaction evidence into these predictions.

A predicted latent has no separately supervised coordinate interpretation as a TCP pose, force vector or physical contact parameter. Use the designated prediction heads rather than interpreting individual latent coordinates as measurements.

## Prediction heads and distribution decoding

Dynamics, reward and each critic receive 451 values: latent 384, action 3 and task context 64. The prior receives 448 values: latent plus task context. Its six outputs parameterize a three-dimensional tanh-squashed Gaussian action distribution, with log standard deviations bounded to [−10, 2].

Reward and each online critic output 101 logits on symlog support [−10,10]. Logits are not physical rewards or values. `decode` applies softmax, computes the weighted support centre and then applies `symexp(x) = sign(x) (exp(abs(x)) − 1)`. The three critics remain separate until the requested aggregation.

`policy(..., deterministic=True)` returns `tanh(mean)` for a convenient demonstration. For MPC evaluation, policy and elite sampling remain active while only extra final exploration noise is removed. Default `policy(...)` samples the prior. Candidate scoring also samples the terminal prior and two online critics, even if the real proposed action came from a deterministic prior call.

## Auxiliary contact supervision

Both auxiliary heads receive the measured contact context **before** latent fusion:

| Output | Shape | Model quantity |
|---|---|---|
| Alignment | `[B,2]` | Held-part lateral displacement in the ground-truth fixed-part mating frame, in metres |
| Force-regime logits | `[B,3]` | Free, nominal contact, high force, in that order |

The alignment label is not the TCP pose or the perturbed runtime target reference. Force labels use plug/socket **pair-force norm** after a causal three-sample median. Thresholds are `<1 N`, `[1,10) N`, and `>=10 N`; startup uses available samples and the lower median for two samples. That label signal differs from the wrist wrench used as model input.

The labels supervise representation learning in simulation; they are not required at execution. Auxiliary outputs are not added to the base controller's candidate score. High-force class predictions are not mechanical-jam decisions; the separate near-jam evaluation diagnostic uses force and progress over a window.

## Candidate scoring

For a supplied action sequence, `rollout` evaluates discounted decoded rewards plus a terminal value. The terminal value averages two randomly selected **online** critics at an action sampled from the prior. It returns all successor latents and a scalar score per candidate.

For example, a batched proposal tensor `[3,B,N,3]` contains horizon 3, batch size B, N candidates and three action components. Expand the current latent and task context to `[B,N,384]` and `[B,N,64]`; `rollout` returns `[3,B,N,384]` successor latents and `[B,N,1]` scores. Selecting a candidate for execution uses its **first** action. The next real measurement then refreshes the latent again.

The MPC evaluation configuration uses horizon 3, discount 0.95, 512 trajectories (24 prior trajectories), six refinement iterations, 64 elites, temperature 0.5 and a 15 Hz control frequency. The candidate example instead uses a small set of uniformly sampled proposals to explain the model interface. It does not implement the full MPPI proposal/refinement procedure. Never cache the terminal imagined latent as the next measured state.

## Component variants and optional calibration

M0 encodes current state and task context without interaction history or recurrent input. M1 adds history, previous latent and preceding action by direct concatenation. M2/M3 use the separate contact encoder and recurrent fusion; they share an inference architecture, while auxiliary supervision distinguishes their learned weights.

Optional target-domain calibration adds a residual correction to frozen latent dynamics. The correction is propagated at each imagined step, without post-add SimNorm and without future measured wrench. The legacy dynamics interface exposes `TaskConditionedLatentContactResidualAdapter` for residual corrections. Connecting a residual to `FACA` requires an explicit integration. Fitting that adapter, simulation training, human-assisted data collection and hardware control are outside this model release.

The runnable examples use randomly initialized weights to demonstrate the inference interfaces. Parameter counts and historical-model compatibility are documented in [the parameter reconciliation](parameter_counts.md).
