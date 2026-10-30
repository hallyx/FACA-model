import pytest
import torch

from faca import (InferenceSession, FACAConfig, FACA, normalize_wrench,
                  task_descriptor)


def small_model(variant='m3'):
    return FACA(FACAConfig(variant=variant, latent_dim=16, hidden_dim=16,
                               prediction_hidden_dim=32, history_dim=8, contact_dim=8))


def data():
    descriptor = task_descriptor(torch.tensor([0, 1]), 7.986, 0.1, 28.616)
    return torch.randn(2, 17), torch.randn(2, 6), descriptor


def test_default_architecture_dimensions():
    model = FACA()
    assert model.task_encoder.fusion[0].in_features == 40
    assert model.history_encoder.net[0].in_features == 104
    assert model.contact_encoder.input_dim == 468
    assert model.state_encoder.input_dim == 532
    assert model.dynamics[0].in_features == 451
    assert model.dynamics[-1].out_features == 384
    assert model.policy_head[0].in_features == 448
    assert len(model.critics) == 3 and model.critics[0][-1].out_features == 101
    assert model.alignment_head[0].in_features == 64
    assert model.force_regime_head[-1].out_features == 3


def test_descriptor_and_wrench_units():
    descriptor = task_descriptor(1, 7.986, 3.887, 28.616)
    torch.testing.assert_close(descriptor, torch.tensor([1., 0., 1., 2*3.887/7.986, 1.]))
    torch.testing.assert_close(normalize_wrench(torch.tensor([100., -50., 0., 10., -5., 0.])),
                               torch.tensor([2., -1., 0., 2., -1., 0.]))
    with pytest.raises(ValueError):
        task_descriptor(0, 0., 1., 1.)


@pytest.mark.parametrize('variant', ['m0', 'm1', 'm2', 'm3'])
def test_session_and_rollout(variant):
    model = small_model(variant).eval()
    state, wrench, descriptor = data()
    session = InferenceSession(model, 2)
    measured = session.observe(state, wrench, descriptor)
    saved = session.previous_latent.clone()
    with torch.no_grad():
        successors, scores = model.rollout(measured.latent, torch.zeros(3, 2, 3), measured.task_context)
    assert successors.shape == (3, 2, 16) and scores.shape == (2, 1)
    assert torch.isfinite(scores).all()
    assert torch.equal(session.previous_latent, saved)
    with pytest.raises(RuntimeError, match='Commit'):
        session.observe(state, wrench, descriptor)
    action = torch.ones(2, 3) * 0.25
    session.commit_action(action)
    session.observe(state, wrench, descriptor, reset_mask=torch.tensor([True, False]))
    assert session.action_history[0].count_nonzero() == 0
    assert torch.equal(session.action_history[1, -1], action[1])
    assert session.state_history[0, :-1].count_nonzero() == 0
    assert torch.equal(session.state_history[:, -1], state)


def test_auxiliary_supervises_contact_before_fusion():
    model = small_model()
    state, wrench, descriptor = data()
    history = {'state_history': state[:, None].expand(-1, 4, -1).clone(),
               'force_history': wrench[:, None].expand(-1, 4, -1).clone(),
               'action_history': torch.zeros(2, 4, 3)}
    measured = model.encode(state, descriptor, history=history)
    alignment, regime = model.auxiliary(measured)
    assert alignment.shape == (2, 2) and regime.shape == (2, 3)
    (alignment.square().sum() + regime.square().sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_encoder.parameters())
    assert all(p.grad is None for p in model.state_encoder.parameters())
    assert all(p.grad is None for p in model.task_encoder.parameters())


def test_task_prior_does_not_enter_contact_encoder_directly():
    model = small_model()
    state, wrench, descriptor = data()
    history = {'state_history': torch.randn(2, 4, 17),
               'force_history': torch.randn(2, 4, 6), 'action_history': torch.randn(2, 4, 3)}
    first = model.encode(state, descriptor, history=history)
    changed = descriptor.clone(); changed[:, 1:] += 1
    second = model.encode(state, changed, history=history)
    torch.testing.assert_close(first.contact_context, second.contact_context, rtol=0, atol=0)
    assert not torch.allclose(first.latent, second.latent)


def test_symlog_decode_and_history_requirement():
    model = small_model()
    decoded = model.decode(torch.zeros(2, 101))
    torch.testing.assert_close(decoded, torch.zeros_like(decoded), atol=1e-6, rtol=0)
    state, _, descriptor = data()
    with pytest.raises(ValueError, match='history'):
        model.encode(state, descriptor)
    with pytest.raises(TypeError):
        model.next(torch.zeros(2, 16), torch.zeros(2, 3), torch.zeros(2, 64), force_history=None)


@pytest.mark.parametrize('variant', ['m0', 'm1', 'm2', 'm3'])
def test_parameter_counts_match_independent_layer_arithmetic(variant):
    from faca.parameter_audit import analytical_module_counts, trainable_count
    for cfg in (FACAConfig(variant=variant), small_model(variant).config):
        model = FACA(cfg)
        counts = analytical_module_counts(cfg)
        assert {name: trainable_count(module) for name, module in model.named_children()} == counts
        assert trainable_count(model) == sum(counts.values())


def test_legacy_counts_and_version_differences():
    from faca.parameter_audit import LEGACY_COUNTS, parameter_report
    report = parameter_report()
    expected_model = {'m0': 1368682, 'm1': 1465130, 'm2': 1550703, 'm3': 1550703}
    for variant, row in report.items():
        assert row['legacy_total'] == LEGACY_COUNTS[variant]
        assert row['model_total'] == expected_model[variant]
    # Reconcile the difference using actual layer changes, not a padding tensor.
    task_change = 18358 - 8400
    state_change = 3 * 128
    history_change = 4 * 3 * 128
    auxiliary_change = 2 * ((384 + 64) - 64) * 128
    assert report['m0']['legacy_minus_current'] == task_change - state_change
    assert report['m1']['legacy_minus_current'] == task_change - state_change - history_change
    assert report['m3']['legacy_minus_current'] == (task_change + auxiliary_change
                                                     - 2 * state_change - history_change)


@pytest.mark.parametrize('variant', ['m0', 'm1'])
def test_baseline_encoder_uses_one_128_wide_hidden_block(variant):
    from torch import nn
    model = FACA(FACAConfig(variant=variant))
    layers = [layer for layer in model.state_encoder if isinstance(layer, nn.Linear)]
    assert [layer.out_features for layer in layers] == [128, 384]


def test_previous_names_are_aliases_of_the_canonical_model():
    from faca import FACA, FACAConfig, PaperFACA, PaperConfig
    from faca.paper import PaperFACA as PreviousModuleModel
    assert PaperFACA is FACA
    assert PreviousModuleModel is FACA
    assert PaperConfig is FACAConfig
