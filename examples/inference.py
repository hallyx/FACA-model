"""Synthetic random-weight inference demonstration, without a simulator.

Examples after installing the package:
    python examples/inference.py
    python examples/inference.py --variant m1 --batch-size 1
    python examples/inference.py --mode candidates --candidates 8
"""
import argparse

import torch

from faca import (InferenceSession, FACAConfig, FACA, normalize_wrench,
                  task_descriptor)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('m0', 'm1', 'm2', 'm3'), default='m3')
    parser.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda:0')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=('prior', 'candidates'), default='prior')
    parser.add_argument('--candidates', type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.candidates) < 1:
        parser.error('batch-size, steps and candidates must be positive')
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = FACAConfig(variant=args.variant)
    model = FACA(cfg).to(device).eval()
    session = InferenceSession(model, batch_size=args.batch_size)
    kind = torch.arange(args.batch_size, device=device) % 2
    descriptor = task_descriptor(kind, 7.986, 0.1, 28.616)
    print(f'variant={args.variant} device={device} mode={args.mode} weights=random inputs=synthetic')
    for step in range(args.steps):
        # Placeholder observations demonstrate tensor flow, not robot behavior.
        state = torch.zeros(args.batch_size, 17, device=device)
        state[:, 3] = 1.0  # quaternion wxyz: identity
        wrench = normalize_wrench(torch.zeros(args.batch_size, 6, device=device))
        reset = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        if step == 2:
            reset[0] = True  # demonstrate a new episode in one batch row
        measured = session.observe(state, wrench, descriptor, reset_mask=reset)
        with torch.no_grad():
            if args.mode == 'prior':
                action, _ = model.policy(measured.latent, measured.task_context,
                                         deterministic=True)
                print(f'step={step} latent={tuple(measured.latent.shape)} '
                      f'action={tuple(action.shape)} reset_rows={reset.nonzero().flatten().tolist()}')
            else:
                # Uniform proposals demonstrate scoring without MPPI refinement.
                n = args.candidates
                latent = measured.latent[:, None].expand(-1, n, -1)
                context = measured.task_context[:, None].expand(-1, n, -1)
                proposals = torch.rand(3, args.batch_size, n, 3, device=device) * 2 - 1
                _, scores = model.rollout(latent, proposals, context)
                best = scores.squeeze(-1).argmax(-1)
                action = proposals[0, torch.arange(args.batch_size, device=device), best]
                print(f'step={step} scores={tuple(scores.shape)} '
                      f'selected_action={tuple(action.shape)} reset_rows={reset.nonzero().flatten().tolist()}')
            if model.contact_encoder is not None:
                alignment, regime_logits = model.auxiliary(measured)
                print(f'  alignment={tuple(alignment.shape)} regime_logits={tuple(regime_logits.shape)}')
        # An integration must commit the action actually executed, after any
        # clipping, replacement or correction in normalized model-action units.
        session.commit_action(action)


if __name__ == '__main__':
    main()
