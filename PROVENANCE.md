# Provenance and scope

Extracted from FACA branch `archive/code-final-20261016`, commit `1574417ec1c352c87e3e1af80edae9b7b6ad235d`. That branch descends from tag `code-final-20261016` (`95b40beac0a2d614a4187f7b168ed141f67154a6`) and the mainline source commit `16c49684103daf596a6e0da5b38180f46867b07c`.

`faca/world_model.py`, `faca/models/`, `faca/init.py`, and `faca/observation_contract.py` derive from the corresponding `tdmpc2` files in that commit, with imports adapted to this package. `faca/layers.py`, `faca/math.py`, and `faca/force_cycle.py` contain selected model dependencies from the corresponding source files. The example config contains model-relevant defaults extracted from the source configuration with smaller dimensions for a CPU demonstration.

FACA originated from the Newt fork `https://github.com/hallyx/Newt`; the original repository retains its upstream history. This independent model-only repository starts a new history to avoid distributing the training code and historical experiment materials. The original MIT notice is reproduced in LICENSE. Some model building blocks document their inspirations in source comments; upstream and third-party licensing and contributor rights should be reviewed by the owner before publication.
