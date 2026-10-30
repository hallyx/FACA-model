"""FACA recurrent world model with separate task and contact pathways.

The model combines nominal task context and measured interaction evidence
for recurrent state updates, latent prediction and action scoring.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .init import weight_init
from .layers import mlp
from .math import gaussian_logprob, log_std, squash
from .models.execution_history_encoder import ExecutionHistoryEncoder
from .models.faca_factorized import BeliefEncoder, ContactEvidenceEncoder, _SimNorm


@dataclass(frozen=True)
class FACAConfig:
    variant: str = 'm3'
    state_dim: int = 17
    action_dim: int = 3
    latent_dim: int = 384
    task_dim: int = 64
    history_len: int = 4
    history_dim: int = 64
    contact_dim: int = 64
    hidden_dim: int = 128
    prediction_hidden_dim: int = 256
    baseline_hidden_dim: int = 128
    num_critics: int = 3
    num_bins: int = 101
    simnorm_dim: int = 8
    vmin: float = -10.0
    vmax: float = 10.0

    def __post_init__(self):
        if self.variant not in ('m0', 'm1', 'm2', 'm3'):
            raise ValueError('variant must be m0, m1, m2 or m3')
        if (self.state_dim, self.action_dim, self.task_dim) != (17, 3, 64):
            raise ValueError('The FACA input contract requires state=17, action=3 and task context=64')
        widths = (self.latent_dim, self.history_len, self.history_dim, self.contact_dim,
                  self.hidden_dim, self.prediction_hidden_dim, self.baseline_hidden_dim, self.num_critics,
                  self.num_bins, self.simnorm_dim)
        if any(value < 1 for value in widths):
            raise ValueError('All dimensions must be positive')
        if self.latent_dim % self.simnorm_dim or self.num_critics < 2 or self.num_bins < 2:
            raise ValueError('Require SimNorm divisibility, at least two critics and two bins')
        if self.vmin >= self.vmax:
            raise ValueError('vmin must be smaller than vmax')


class TaskDescriptorEncoder(nn.Module):
    """Task encoding: type embedding 2x8, metrics 4->32->32, fusion 40->64->64."""

    def __init__(self):
        super().__init__()
        self.type_embedding = nn.Embedding(2, 8)
        self.metrics = mlp(4, [32], 32, act=nn.Mish())
        self.fusion = mlp(40, [64], 64, act=nn.Mish())

    def forward(self, descriptor):
        if descriptor.shape[-1] != 5 or not torch.isfinite(descriptor).all():
            raise ValueError('descriptor must contain five finite values')
        kind = descriptor[..., 0]
        if not ((kind == 0) | (kind == 1)).all():
            raise ValueError('mating type must be 0 (peg-in-hole) or 1 (hole-onto-peg)')
        return self.fusion(torch.cat((self.type_embedding(kind.long()),
                                      self.metrics(descriptor[..., 1:])), dim=-1))


@dataclass
class MeasuredState:
    latent: torch.Tensor
    task_context: torch.Tensor
    contact_context: torch.Tensor | None


class FACA(nn.Module):
    """Measured update, prediction heads and policy prior; defaults use a 384D latent.

    M2 and M3 share this inference architecture. Their distinction is in training
    supervision, which is outside the scope of this package.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or FACAConfig()
        cfg = self.config
        self.task_encoder = TaskDescriptorEncoder()
        self.history_encoder = None
        self.contact_encoder = None
        self.alignment_head = None
        self.force_regime_head = None
        if cfg.variant != 'm0':
            self.history_encoder = ExecutionHistoryEncoder(
                cfg.history_len, cfg.history_dim, state_dim=cfg.state_dim,
                force_dim=6, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim)
        if cfg.variant in ('m2', 'm3'):
            self.contact_encoder = ContactEvidenceEncoder(
                state_dim=cfg.state_dim, history_context_dim=cfg.history_dim,
                latent_dim=cfg.latent_dim, action_dim=cfg.action_dim,
                contact_context_dim=cfg.contact_dim, hidden_dim=cfg.hidden_dim)
            self.state_encoder = BeliefEncoder(
                state_dim=cfg.state_dim, latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim, task_dim=cfg.task_dim,
                contact_context_dim=cfg.contact_dim, hidden_dim=cfg.hidden_dim,
                simnorm_dim=cfg.simnorm_dim)
            self.alignment_head = mlp(cfg.contact_dim, [cfg.hidden_dim], 2)
            self.force_regime_head = mlp(cfg.contact_dim, [cfg.hidden_dim], 3)
        else:
            input_dim = cfg.state_dim + cfg.task_dim
            if cfg.variant == 'm1':
                input_dim += cfg.history_dim + cfg.latent_dim + cfg.action_dim
            self.state_encoder = mlp(input_dim, [cfg.baseline_hidden_dim],
                                     cfg.latent_dim, act=_SimNorm(cfg.simnorm_dim))
        prediction_input = cfg.latent_dim + cfg.action_dim + cfg.task_dim
        hidden = [cfg.prediction_hidden_dim] * 2
        self.dynamics = mlp(prediction_input, hidden, cfg.latent_dim,
                            act=_SimNorm(cfg.simnorm_dim))
        self.reward_head = mlp(prediction_input, hidden, cfg.num_bins)
        self.critics = nn.ModuleList(mlp(prediction_input, hidden, cfg.num_bins)
                                     for _ in range(cfg.num_critics))
        self.policy_head = mlp(cfg.latent_dim + cfg.task_dim, hidden, 2 * cfg.action_dim)
        self.register_buffer('bin_centres', torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins))
        self.apply(weight_init)
        nn.init.zeros_(self.reward_head[-1].weight)
        for critic in self.critics:
            nn.init.zeros_(critic[-1].weight)

    def encode(self, state, descriptor, *, history=None, previous_latent=None,
               previous_action=None):
        """Update measured latent state. All histories are oldest-first (s_i, w_i, a_{i-1})."""
        cfg = self.config
        if state.shape[-1] != cfg.state_dim:
            raise ValueError('state must be 17D; wrench is a separate history channel')
        context = self.task_encoder(descriptor)
        if state.shape[:-1] != context.shape[:-1]:
            raise ValueError('state and descriptor batch dimensions must match')
        if cfg.variant == 'm0':
            return MeasuredState(self.state_encoder(torch.cat((state, context), -1)), context, None)
        if history is None:
            raise ValueError('M1-M3 require explicit execution history')
        feature = self.history_encoder(history['state_history'], history['force_history'],
                                       history['action_history'])
        if feature.shape[:-1] != state.shape[:-1]:
            raise ValueError('history and state batch dimensions must match')
        if previous_latent is None:
            previous_latent = state.new_zeros(*state.shape[:-1], cfg.latent_dim)
        if previous_action is None:
            previous_action = state.new_zeros(*state.shape[:-1], cfg.action_dim)
        if self.contact_encoder is not None:
            contact = self.contact_encoder(state, feature, previous_latent, previous_action)
            latent = self.state_encoder(state, previous_latent, previous_action, context, contact)
            return MeasuredState(latent, context, contact)
        latent = self.state_encoder(torch.cat((state, context, feature,
                                                previous_latent, previous_action), -1))
        return MeasuredState(latent, context, None)

    @staticmethod
    def _prediction_input(latent, action, task_context):
        return torch.cat((latent, action, task_context), dim=-1)

    def next(self, latent, action, task_context):
        """Predict the next latent from latent, action and static task context."""
        return self.dynamics(self._prediction_input(latent, action, task_context))

    def reward_logits(self, latent, action, task_context):
        return self.reward_head(self._prediction_input(latent, action, task_context))

    def value_logits(self, latent, action, task_context):
        x = self._prediction_input(latent, action, task_context)
        return torch.stack([critic(x) for critic in self.critics])

    def decode(self, logits):
        symlog_value = (logits.softmax(-1) * self.bin_centres).sum(-1, keepdim=True)
        return symlog_value.sign() * symlog_value.abs().expm1()

    def policy(self, latent, task_context, *, deterministic=False):
        """Return action and log density; deterministic gives tanh(mean)."""
        mean, raw_std = self.policy_head(torch.cat((latent, task_context), -1)).chunk(2, -1)
        std_log = log_std(raw_std, -10.0, 12.0)
        if deterministic:
            return mean.tanh(), None
        noise = torch.randn_like(mean)
        sample = mean + noise * std_log.exp()
        density = gaussian_logprob(noise, std_log)
        _, action, density = squash(mean, sample, density)
        return action, density

    def auxiliary(self, measured):
        """Predict alignment and force regime from contact context before fusion."""
        if self.alignment_head is None or measured.contact_context is None:
            raise ValueError('Contact auxiliary heads are available only in M2/M3')
        return (self.alignment_head(measured.contact_context),
                self.force_regime_head(measured.contact_context))

    def rollout(self, latent, actions, task_context, *, discount=0.95):
        """Score candidate sequences with decoded rewards and a terminal value.

        actions has shape (horizon, ..., 3). Return successor latents and scores.
        Terminal value averages two randomly selected ONLINE critics at a sampled
        prior action. MPPI proposal refinement is outside this method.
        """
        if actions.shape[0] < 1:
            raise ValueError('rollout horizon must be positive')
        score = latent.new_zeros(*latent.shape[:-1], 1)
        successors = []
        for step, action in enumerate(actions):
            score = score + discount ** step * self.decode(
                self.reward_logits(latent, action, task_context))
            latent = self.next(latent, action, task_context)
            successors.append(latent)
        terminal_action, _ = self.policy(latent, task_context)
        indices = torch.randperm(self.config.num_critics, device=latent.device)[:2]
        terminal_value = self.decode(self.value_logits(latent, terminal_action, task_context)[indices]).mean(0)
        score = score + discount ** len(actions) * terminal_value
        return torch.stack(successors), score
