from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .observation_contract import policy_state_dim

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation.
	"""

	def __init__(self, *args, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act

	def forward(self, x):
		x = super().forward(x)
		return self.act(self.ln(x))

	def __repr__(self):
		if isinstance(self.act, nn.Sequential):
			# print the name of each module in the sequential
			act = '[' + ', '.join([m.__class__.__name__ for m in self.act]) + ']'
		else:
			act = self.act.__class__.__name__
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}, "\
			f"act={act})"


def mlp(in_dim, mlp_dims, out_dim, act=None):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1]))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class QEnsemble(nn.Module):
	"""
	Vectorized ensemble of Q-networks. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		mlp_dims = 2*[cfg.mlp_dim]
		out_dim = max(cfg.num_bins, 1)
		self._Qs = nn.ModuleList([mlp(in_dim, mlp_dims, out_dim) for _ in range(cfg.num_q)])
		if cfg.compile:
			if cfg.rank == 0:
				print('Compiling QEnsemble forward...')
			self._forward = torch.compile(self._forward_impl, mode='reduce-overhead')
		else:
			self._forward = self._forward_impl

	def _forward_impl(self, x):
		outs = [q(x) for q in self._Qs]
		return torch.stack(outs, dim=0)

	def forward(self, x):
		return self._forward(x)


class QOnlineTargetEnsemble(nn.Module):
	"""
	Online and target Q-ensembles for TD-MPC2. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.online = QEnsemble(cfg)
		self.target = deepcopy(self.online)
		self.tau = cfg.tau
		self.target.train(False)
		self.track_grad(False, network='target')

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		self.online.train(mode)
		self.target.train(False)
		return self

	def track_grad(self, mode=True, network='online'):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		assert network in {'online', 'target'}
		module = self.online if network == 'online' else self.target
		for p in module.parameters():
			p.requires_grad_(mode)

	@torch.no_grad()
	def hard_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.copy_(op.data)

	@torch.no_grad()
	def soft_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.lerp_(op.data, self.tau)

	def forward(self, x, target=False):
		if target:
			return self.target(x)
		else:
			return self.online(x)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if cfg.obs == 'state':
		state_input_dim = cfg.obs_shape['state'][0] + cfg.task_dim
		if str(cfg.get('faca_variant', 'none')).lower() == 'm1':
			state_input_dim = (
				policy_state_dim(cfg) +
				int(cfg.task_dim) +
				int(cfg.get('faca_history_context_dim', 64)) +
				int(cfg.latent_dim) +
				int(cfg.action_dim)
			)
		cycle_variant = cfg.get('force_cycle_variant', 'none')
		if cycle_variant in {'f1','f2','f3'}:
			from .force_cycle import CycleEncoder
			if cfg.obs_shape['state'][0] != 62: raise ValueError('Force-cycle observation must be 62D')
			out['state'] = CycleEncoder(cfg, mlp(14+64+cfg.task_dim, max(cfg.num_enc_layers-1,1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg)))
		else:
			out['state'] = mlp(state_input_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	elif cfg.obs == 'rgb':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim + cfg.obs_shape['rgb'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	# for k in cfg.obs_shape.keys():
	# 	if k == 'state':
	# 		out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	# 	# elif k == 'rgb':
	# 	# 	out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
	# 	elif k == 'rgb':
	# 		# out[k] = nn.Sequential(
	# 		# 	nn.Linear(cfg.latent_dim + cfg.task_dim + cfg.obs_shape[k][0], cfg.latent_dim),
	# 		# 	SimNorm(cfg)
	# 		# )
	# 		# out[k] = nn.Sequential(
	# 		# 	nn.Linear(cfg.obs_shape[k][0], cfg.latent_dim),
	# 		# 	SimNorm(cfg)
	# 		# )
	# 		# out['fuse'] = nn.Sequential(
	# 		# 	nn.Linear(2*cfg.latent_dim, cfg.latent_dim),
	# 		# 	SimNorm(cfg),
	# 		# )
	# 	else:
	# 		raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)
