"""Batched measured-state history and recurrence for FACA inference."""
import torch


class InferenceSession:
    """Refresh from real measurements and cache only measured latents/actions.

    Call observe at reset and after every executed action, then commit_action
    with the action actually executed. Reset rows using reset_mask on observe.
    Wrench inputs must already be normalized. A session belongs to one model,
    batch size and device; construct it after moving the model to its device.
    """

    def __init__(self, model, batch_size):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.model = model
        cfg = model.config
        reference = next(model.parameters())
        self.state_history = reference.new_zeros(batch_size, cfg.history_len, cfg.state_dim)
        self.force_history = reference.new_zeros(batch_size, cfg.history_len, 6)
        self.action_history = reference.new_zeros(batch_size, cfg.history_len, cfg.action_dim)
        self.previous_latent = reference.new_zeros(batch_size, cfg.latent_dim)
        self.previous_action = reference.new_zeros(batch_size, cfg.action_dim)
        self.awaiting_action = False
        self.batch_size = batch_size
        self.descriptor = None

    @torch.no_grad()
    def observe(self, state, normalized_wrench, descriptor, *, reset_mask=None):
        if self.awaiting_action:
            raise RuntimeError('Commit the executed action before the next observation')
        cfg = self.model.config
        if state.shape != (self.batch_size, cfg.state_dim):
            raise ValueError('State shape does not match the session')
        if normalized_wrench.shape != (self.batch_size, 6) or descriptor.shape != (self.batch_size, 5):
            raise ValueError('Expected batched wrench [B,6] and descriptor [B,5]')
        reset = torch.zeros(self.batch_size, dtype=torch.bool, device=state.device)
        if reset_mask is not None:
            reset = torch.as_tensor(reset_mask, dtype=torch.bool, device=state.device)
            if reset.shape != (self.batch_size,):
                raise ValueError('reset_mask must have shape [B]')
        if self.descriptor is not None:
            changed = (descriptor != self.descriptor).any(-1)
            if (changed & ~reset).any():
                raise ValueError('Descriptor must be fixed within an episode; reset changed rows')
        for value in (self.state_history, self.force_history, self.action_history,
                      self.previous_latent, self.previous_action):
            value[reset] = 0
        for history, current in ((self.state_history, state),
                                 (self.force_history, normalized_wrench),
                                 (self.action_history, self.previous_action)):
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = current
        measured = self.model.encode(state, descriptor,
            history={'state_history': self.state_history,
                     'force_history': self.force_history,
                     'action_history': self.action_history},
            previous_latent=self.previous_latent, previous_action=self.previous_action)
        self.previous_latent.copy_(measured.latent)
        self.descriptor = descriptor.detach().clone()
        self.awaiting_action = True
        return measured

    @torch.no_grad()
    def commit_action(self, executed_action):
        if not self.awaiting_action:
            raise RuntimeError('Observe before committing an action')
        if executed_action.shape != self.previous_action.shape:
            raise ValueError('Executed action must have shape [B,3]')
        self.previous_action.copy_(executed_action)
        self.awaiting_action = False
