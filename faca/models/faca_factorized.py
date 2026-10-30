"""FACA contact evidence, recurrent state fusion and auxiliary prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NormedHidden(nn.Module):
	def __init__(self, in_dim: int, out_dim: int):
		super().__init__()
		self.linear = nn.Linear(int(in_dim), int(out_dim))
		self.norm = nn.LayerNorm(int(out_dim))
		self.act = nn.Mish(inplace=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(self.norm(self.linear(x)))


class _SimNorm(nn.Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = int(dim)
		if self.dim < 1:
			raise ValueError(f"simnorm_dim must be positive, got {self.dim}.")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		shape = x.shape
		if int(shape[-1]) % self.dim != 0:
			raise ValueError(
				f"Feature width {shape[-1]} is not divisible by simnorm_dim={self.dim}."
			)
		return F.softmax(x.reshape(*shape[:-1], -1, self.dim), dim=-1).reshape(*shape)


class ContactEvidenceEncoder(nn.Module):
	"""Encode observed contact evidence without exposing it to imagination."""

	def __init__(
		self,
		*,
		state_dim: int,
		history_context_dim: int,
		latent_dim: int,
		action_dim: int,
		contact_context_dim: int = 64,
		hidden_dim: int = 128,
	):
		super().__init__()
		self.state_dim = int(state_dim)
		self.history_context_dim = int(history_context_dim)
		self.latent_dim = int(latent_dim)
		self.action_dim = int(action_dim)
		self.contact_context_dim = int(contact_context_dim)
		self.input_dim = (
			self.state_dim + self.history_context_dim + self.latent_dim + self.action_dim
		)
		self.net = nn.Sequential(
			_NormedHidden(self.input_dim, int(hidden_dim)),
			nn.Linear(int(hidden_dim), self.contact_context_dim),
		)

	def forward(
		self,
		state: torch.Tensor,
		history_context: torch.Tensor,
		prev_z: torch.Tensor,
		prev_action: torch.Tensor,
	) -> torch.Tensor:
		x = torch.cat([state, history_context, prev_z, prev_action], dim=-1)
		if int(x.shape[-1]) != self.input_dim:
			raise ValueError(
				f"ContactEvidenceEncoder expected width {self.input_dim}, got {tuple(x.shape)}."
			)
		return self.net(x)


class BeliefEncoder(nn.Module):
	"""Recurrent factorized belief fusion from current, causal evidence only."""

	def __init__(
		self,
		*,
		state_dim: int,
		latent_dim: int,
		action_dim: int,
		task_dim: int,
		contact_context_dim: int,
		hidden_dim: int = 128,
		simnorm_dim: int = 8,
	):
		super().__init__()
		self.input_dim = (
			int(state_dim) + int(latent_dim) + int(action_dim) + int(task_dim) +
			int(contact_context_dim)
		)
		self.latent_dim = int(latent_dim)
		self.net = nn.Sequential(
			_NormedHidden(self.input_dim, int(hidden_dim)),
			nn.Linear(int(hidden_dim), self.latent_dim),
			_SimNorm(int(simnorm_dim)),
		)

	def forward(
		self,
		state: torch.Tensor,
		prev_z: torch.Tensor,
		prev_action: torch.Tensor,
		task_context: torch.Tensor,
		contact_context: torch.Tensor,
	) -> torch.Tensor:
		x = torch.cat([state, prev_z, prev_action, task_context, contact_context], dim=-1)
		if int(x.shape[-1]) != self.input_dim:
			raise ValueError(f"BeliefEncoder expected width {self.input_dim}, got {tuple(x.shape)}.")
		return self.net(x)


class AlignmentHead(nn.Module):
	def __init__(self, latent_dim: int, task_dim: int, hidden_dim: int = 128):
		super().__init__()
		self.input_dim = int(latent_dim) + int(task_dim)
		self.net = nn.Sequential(
			_NormedHidden(self.input_dim, int(hidden_dim)),
			nn.Linear(int(hidden_dim), 2),
		)

	def forward(self, z: torch.Tensor, task_context: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([z, task_context], dim=-1))


class ContactModeHead(nn.Module):
	def __init__(self, latent_dim: int, task_dim: int, hidden_dim: int = 128):
		super().__init__()
		self.input_dim = int(latent_dim) + int(task_dim)
		self.net = nn.Sequential(
			_NormedHidden(self.input_dim, int(hidden_dim)),
			nn.Linear(int(hidden_dim), 3),
		)

	def forward(self, z: torch.Tensor, task_context: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([z, task_context], dim=-1))
