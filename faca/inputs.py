"""FACA input construction, without environment or simulator dependencies."""
import torch


def task_descriptor(mating_type, width_mm, clearance_mm, axial_extent_mm):
    """Construct the five-value descriptor using millimetres and fixed reference scales.

    Returns [..., 5]: type, log(D/7.986), c/3.887, 2c/D, d/28.616.
    Clearance is the configured descriptor, not a measured surface gap.
    """
    values = [torch.as_tensor(v) for v in
              (mating_type, width_mm, clearance_mm, axial_extent_mm)]
    device = next((v.device for v in values if v.device.type != 'cpu'), values[0].device)
    kind, width, clearance, extent = torch.broadcast_tensors(
        *(v.to(device=device, dtype=torch.float32) for v in values))
    if not all(torch.isfinite(v).all() for v in (kind, width, clearance, extent)):
        raise ValueError('Descriptor inputs must be finite')
    if not ((kind == 0) | (kind == 1)).all():
        raise ValueError('Mating type must be 0 or 1')
    if (width <= 0).any() or (extent <= 0).any() or (clearance < 0).any():
        raise ValueError('Width and extent must be positive, clearance nonnegative')
    return torch.stack((kind, (width / 7.986).log(), clearance / 3.887,
                        2 * clearance / width, extent / 28.616), -1)


def normalize_wrench(tared_tcp_wrench):
    """Scale an already tared, TCP-framed wrench [N,N,N,Nm,Nm,Nm].

    Applies no clipping or temporal filter. This function does not perform sensor
    calibration or a frame transform and must not receive plug/socket pair force.
    """
    if tared_tcp_wrench.shape[-1] != 6:
        raise ValueError('Wrench must be six-dimensional')
    return tared_tcp_wrench / tared_tcp_wrench.new_tensor([50., 50., 50., 5., 5., 5.])
