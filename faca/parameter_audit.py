"""Count FACA model parameters and verify current and legacy configurations.

Run: python -m faca.parameter_audit
No checkpoints, simulator, training data or optimizer are required.
"""
import contextlib
import io
import json

from .model import FACAConfig, FACA


LEGACY_COUNTS = {'m0': 1378256, 'm1': 1473168, 'm2': 1656661, 'm3': 1656661}


def historical_count_config(variant):
    """Legacy model settings used for parameter-count verification.

    This is the LEGACY 14D/20D observation contract with a six-value descriptor,
    14D history and latent+task auxiliary heads.
    """
    from .example_config import example_config
    if variant not in LEGACY_COUNTS:
        raise ValueError('variant must be m0, m1, m2 or m3')
    cfg = example_config(variant)
    cfg.faca_observation_contract = 'legacy'
    cfg.obs_shape = {'state': [14 if variant == 'm0' else 20]}
    cfg.latent_dim = 384
    cfg.mlp_dim = 256
    cfg.enc_dim = 128
    cfg.num_enc_layers = 2
    cfg.num_bins = 101
    cfg.num_q = 3
    cfg.task_dim = 64
    cfg.axial_task_vec_dim = 6
    cfg.faca_history_len = 4
    cfg.faca_history_state_dim = 14
    cfg.faca_history_force_dim = 6
    cfg.faca_history_action_dim = 3
    cfg.faca_history_layers = 2
    cfg.faca_contact_context_dim = 64
    cfg.faca_contact_hidden_dim = 128
    cfg.faca_belief_hidden_dim = 128
    cfg.faca_aux_head_hidden_dim = 128
    cfg.faca_history_context_dim = 64
    cfg.faca_history_hidden_dim = 128
    return cfg


def trainable_count(module):
    """Count unique parameters requiring gradients; buffers are excluded."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def analytical_module_counts(config):
    """Independent parameter arithmetic for each FACA module.

    An affine layer has (input+1)*output parameters. Hidden LayerNorm adds
    2*output; Mish and SimNorm have none. Task branch outputs are hidden blocks;
    history/contact/auxiliary outputs are plain affine layers. The inherited
    dynamics and baseline output layers have LayerNorm before SimNorm.
    """
    cfg = config

    def linear(a, b):
        return (a + 1) * b

    def normed(a, b):
        return linear(a, b) + 2 * b

    def mlp(a, hidden, b, normalized_output=False):
        result = 0
        for width in hidden:
            result += normed(a, width)
            a = width
        return result + (normed(a, b) if normalized_output else linear(a, b))

    s, z, t, a = cfg.state_dim, cfg.latent_dim, cfg.task_dim, cfg.action_dim
    h, c, width = cfg.history_dim, cfg.contact_dim, cfg.hidden_dim
    prediction_input = z + a + t
    counts = {'task_encoder': 2 * 8 + normed(4, 32) + normed(32, 32)
              + normed(40, 64) + normed(64, 64)}
    if cfg.variant != 'm0':
        counts['history_encoder'] = mlp(cfg.history_len * (s + 6 + a), [width, width], h)
    if cfg.variant in ('m2', 'm3'):
        counts['contact_encoder'] = mlp(s + h + z + a, [width], c)
        counts['state_encoder'] = mlp(s + z + a + t + c, [width], z)
        counts['alignment_head'] = mlp(c, [width], 2)
        counts['force_regime_head'] = mlp(c, [width], 3)
    else:
        state_input = s + t + (h + z + a if cfg.variant == 'm1' else 0)
        counts['state_encoder'] = mlp(state_input, [cfg.baseline_hidden_dim], z, True)
    hidden = [cfg.prediction_hidden_dim] * 2
    counts['dynamics'] = mlp(prediction_input, hidden, z, True)
    counts['reward_head'] = mlp(prediction_input, hidden, cfg.num_bins)
    counts['critics'] = cfg.num_critics * mlp(prediction_input, hidden, cfg.num_bins)
    counts['policy_head'] = mlp(z + t, hidden, 2 * a)
    return counts


def parameter_report():
    from .world_model import WorldModel
    report = {}
    for variant, expected_legacy in LEGACY_COUNTS.items():
        model = FACA(FACAConfig(variant=variant))
        actual = {name: trainable_count(module) for name, module in model.named_children()}
        analytic = analytical_module_counts(model.config)
        if actual != analytic:
            raise AssertionError(f'Layer arithmetic does not match instantiated {variant}: {actual} != {analytic}')
        # The archive constructor prints architecture notices, not audit JSON.
        with contextlib.redirect_stdout(io.StringIO()):
            archived = WorldModel(historical_count_config(variant))
        historical = trainable_count(archived)
        if historical != expected_legacy:
            raise AssertionError(f'Historical {variant} count {historical} != expected {expected_legacy}')
        historical_modules = {name: trainable_count(module)
                              for name, module in archived.named_children()}
        total = sum(actual.values())
        report[variant] = {
            'model_modules': actual,
            'model_total': total,
            'legacy_expected_total': expected_legacy,
            'legacy_total': historical,
            'legacy_modules': historical_modules,
            'legacy_minus_current': expected_legacy - total,
        }
    return report


if __name__ == '__main__':
    print(json.dumps(parameter_report(), indent=2))
