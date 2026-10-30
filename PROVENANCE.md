# Development lineage and attribution

## Upstream foundation

FACA builds on Newt/TD-MPC2. The development history originates from the Newt fork at `https://github.com/hallyx/Newt`. The source repository retains the upstream commit graph, and this model release preserves the MIT license and source-level references for inherited building blocks.

## Release lineage

The model release starts from FACA branch `archive/code-final-20261016`, commit `1574417ec1c352c87e3e1af80edae9b7b6ad235d`. Its predecessors include the frozen tag `code-final-20261016` at `95b40beac0a2d614a4187f7b168ed141f67154a6` and mainline commit `16c49684103daf596a6e0da5b38180f46867b07c`.

This repository has an independent model-release history. Training jobs, replay buffers and experiment outputs remain in the development repository.

## Model components

- `faca/model.py` defines `FACA` and `FACAConfig`: task encoding, contact-aware recurrent state updates, dynamics, reward/value heads and the policy prior.
- `faca/inputs.py` defines the nominal descriptor and wrist-wrench scaling.
- `faca/runtime.py` manages execution history and measured-state recurrence.
- `faca/world_model.py` and the legacy model components preserve earlier interfaces for version compatibility.
- `faca/layers.py`, `faca/math.py`, `faca/init.py` and `faca/models/` contain shared neural building blocks developed in the FACA codebase and its upstream foundation.

The current five-value task interface and the legacy six-value interface have different observation histories and auxiliary-head connections. [Model architecture](docs/architecture.md) records these version boundaries. [Parameter counts](docs/parameter_counts.md) records both configurations and their exact layer differences. Benchmark results belong to their associated checkpoints and evaluation configurations, rather than to randomly initialized examples.

## Research and media artifacts

The project research PDF is dated 2026-10-13, with SHA-256 `fbf281862766e6b310ffcf64d6eada3948fb64d18756bdee2ae9b751754f445f`. It contains parameter totals for the legacy configuration alongside the five-value-interface architecture description; both configurations are documented in the parameter-count record.

The system overview PNG in `docs/assets/faca_architecture.png` is rendered from the project's original Figure 1 PDF, `docs/1.pdf`. Simulation clips in `docs/assets/videos/` are web copies of the files in `videos/`; posters preserve the original frame colors.
