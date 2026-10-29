"""Versioned force-free policy observations with a separate history sensor channel."""
import torch
from tensordict import TensorDict

STATE_GOAL_CONTRACT = 'state_goal_wrench_history_v1'
LEGACY_CONTRACT = 'legacy'


def uses_state_goal(cfg):
    return cfg.get('faca_observation_contract', LEGACY_CONTRACT) == STATE_GOAL_CONTRACT


def policy_state_dim(cfg):
    return 17 if uses_state_goal(cfg) else int(cfg.get('faca_history_state_dim', 14))


def split_state_goal_observation(obs, *, require_wrench=True):
    if not isinstance(obs, (dict, TensorDict)):
        raise ValueError('17D state/goal observations require named state and wrist_wrench channels; legacy tensors cannot be relabeled.')
    state = obs.get('state')
    wrench = obs.get('wrist_wrench')
    if not torch.is_tensor(state) or state.shape[-1] != 17:
        raise ValueError('Policy state must be exactly 17D: 14D canonical state + 3D goal TCP position in socket frame.')
    if require_wrench and (not torch.is_tensor(wrench) or wrench.shape != (*state.shape[:-1], 6)):
        raise ValueError('History wrist_wrench must be a separate 6D tensor aligned with state.')
    return state, wrench


def observation_metadata(cfg):
    if not uses_state_goal(cfg):
        return {'version': LEGACY_CONTRACT}
    return {
        'version': STATE_GOAL_CONTRACT,
        'policy_state_dim': 17,
        'state_layout': ['tcp_pos_socket:3', 'tcp_quat_socket_wxyz:4', 'tcp_linvel_socket:3',
                         'tcp_angvel_socket:3', 'gripper_width:1', 'goal_tcp_pos_socket:3'],
        'goal_source': 'gripper_goal_pos',
        'goal_frame': 'socket_observation_frame',
        'goal_units': 'm',
        'wrench_channel': 'wrist_wrench',
        'wrench_route': 'execution_history_only' if cfg.get('faca_variant') != 'm0' else 'disabled',
        'history_state_dim': 14,
    }


def validate_observation_metadata(cfg, metadata, context):
    expected = observation_metadata(cfg)
    actual = metadata.get('observation_contract', {'version': LEGACY_CONTRACT})
    if actual != expected:
        raise ValueError(f'Incompatible observation contract in {context}: stored={actual!r}, expected={expected!r}. Recollect data; do not relabel legacy observations.')
