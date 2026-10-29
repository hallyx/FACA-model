import pytest
import torch
from tensordict import TensorDict

from faca import WorldModel
from faca.example_config import example_config


def inputs():
    obs = TensorDict({'state': torch.randn(2, 17), 'wrist_wrench': torch.randn(2, 6)}, batch_size=[2])
    history = {'state_history': torch.zeros(2, 4, 14), 'force_history': torch.zeros(2, 4, 6), 'action_history': torch.zeros(2, 4, 3)}
    return obs, history, torch.zeros(2, 6)


@pytest.mark.parametrize('variant', ['m0', 'm1', 'm2', 'm3'])
def test_state_goal_model_forward(variant):
    torch.manual_seed(4)
    model = WorldModel(example_config(variant)).eval()
    obs, history, task = inputs()
    with torch.no_grad():
        z = model.encode(obs, task, history, torch.zeros(2, 16), torch.zeros(2, 3))
        action, _ = model.pi(z, task)
        assert z.shape == (2, 16)
        assert action.shape == (2, 3)
        assert torch.isfinite(z).all() and torch.isfinite(action).all()
        assert model.next(z, action, task).shape == (2, 16)
        assert model.reward(z, action, task).shape == (2, 11)
        if variant in ('m2', 'm3'):
            alignment, contact = model.predict_auxiliary(z, task)
            assert alignment.shape == (2, 2) and contact.shape == (2, 3)


def test_named_wrench_is_history_only():
    torch.manual_seed(3)
    model = WorldModel(example_config('m1')).eval()
    obs, history, task = inputs()
    previous_z, previous_action = torch.zeros(2, 16), torch.zeros(2, 3)
    z = model.encode(obs, task, history, previous_z, previous_action)
    changed = obs.clone()
    changed['wrist_wrench'] += 100
    assert torch.equal(z, model.encode(changed, task, history, previous_z, previous_action))
    history['force_history'][:, -1] = 10
    assert not torch.allclose(z, model.encode(obs, task, history, previous_z, previous_action))
    with pytest.raises(ValueError, match='named state'):
        model.encode(obs['state'], task, history, previous_z, previous_action)
