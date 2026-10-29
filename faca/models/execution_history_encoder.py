import torch
import torch.nn as nn


class ExecutionHistoryEncoder(nn.Module):
	"""Encode the shared FACA execution history ``(o, a, f)``.

	All FACA history variants use the same tensor contract. ``state_history``
	contains the canonical force-free TCP state, ``force_history`` contains the
	normalized wrist wrench, and ``action_history`` contains the actions that
	precede the current observation. Histories are left-padded with zeros at an
	episode or replay-window boundary.
	"""

	def __init__(
		self,
		history_len: int,
		context_dim: int,
		state_dim: int = 14,
		force_dim: int = 6,
		action_dim: int = 3,
		hidden_dim: int = 128,
		num_layers: int = 2,
	):
		super().__init__()
		self.history_len = int(history_len)
		self.context_dim = int(context_dim)
		self.state_dim = int(state_dim)
		self.force_dim = int(force_dim)
		self.action_dim = int(action_dim)
		self.hidden_dim = int(hidden_dim)
		self.num_layers = int(num_layers)
		for name, value in (
			("history_len", self.history_len),
			("context_dim", self.context_dim),
			("state_dim", self.state_dim),
			("force_dim", self.force_dim),
			("action_dim", self.action_dim),
			("hidden_dim", self.hidden_dim),
			("num_layers", self.num_layers),
		):
			if value < 1:
				raise ValueError(f"{name} must be >= 1, got {value}.")

		input_dim = self.history_len * (self.state_dim + self.force_dim + self.action_dim)
		layers = []
		in_dim = input_dim
		for _ in range(self.num_layers):
			layers.extend([
				nn.Linear(in_dim, self.hidden_dim),
				nn.LayerNorm(self.hidden_dim),
				nn.Mish(inplace=False),
			])
			in_dim = self.hidden_dim
		layers.append(nn.Linear(in_dim, self.context_dim))
		self.net = nn.Sequential(*layers)

	def _check_history(self, value, name, feature_dim, batch_shape=None):
		if value is None:
			raise ValueError(f"{name} is required.")
		if not torch.is_tensor(value):
			value = torch.as_tensor(value)
		expected_tail = (self.history_len, feature_dim)
		if value.ndim < 3 or tuple(value.shape[-2:]) != expected_tail:
			raise ValueError(
				f"{name} must have shape (..., {self.history_len}, {feature_dim}), "
				f"got {tuple(value.shape)}."
			)
		if batch_shape is not None and tuple(value.shape[:-2]) != tuple(batch_shape):
			raise ValueError(
				f"{name} batch shape {tuple(value.shape[:-2])} does not match {tuple(batch_shape)}."
			)
		return value.to(dtype=torch.float32)

	def forward(self, state_history, force_history, action_history):
		state_history = self._check_history(
			state_history, "state_history", self.state_dim
		)
		batch_shape = state_history.shape[:-2]
		force_history = self._check_history(
			force_history, "force_history", self.force_dim, batch_shape
		).to(device=state_history.device)
		action_history = self._check_history(
			action_history, "action_history", self.action_dim, batch_shape
		).to(device=state_history.device)
		x = torch.cat([state_history, force_history, action_history], dim=-1)
		return self.net(x.reshape(*batch_shape, -1))
