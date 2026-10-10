import torch as th
import torch.nn.functional as F


def is_multiple(a, b, tolerance=1e-9):
    """
    Check if 'a' is a multiple of 'b', accounting for floating-point errors.
    tolerance: The acceptable error range due to floating-point precision.
    return: True if 'a' is a multiple of 'b', False otherwise.
    """
    if b == 0:  # Avoid division by zero
        return False
    remainder = abs(a % b)
    return remainder < tolerance or abs(remainder - abs(b)) < tolerance


def is_rotation_matrix(R: th.Tensor):
    """
    Check if Rs are valid rotation matrices
    Return th.Tensor (N,) with 1s for valid and 0s for invalid matrices
    """
    if R.shape != (3, 3):
        return False

    # check if R is orthogonal (R.T * R = I)


def safe_atan2(ys, xs, epsilon=1e-10):
    """
    computes atan2(y/x) but ensures no nans in forward or backward pass
    https://discuss.pytorch.org/t/how-to-avoid-nan-output-from-atan2-during-backward-pass/176890/2
    """
    # add epsilon to denominator to avoid NaN in backward pass
    near_zeros = xs.abs() < epsilon
    xs = xs * (near_zeros.logical_not())
    xs = xs + (near_zeros * epsilon)
    return th.atan2(ys, xs)


def smooth_l1(x):
    """
    Smoothed l1 norm
    :param x: n-dim vector
    :return: scalar
    """
    delta = 1.0
    abs_errors = th.norm(x + 1e-6)
    quadratic = th.minimum(abs_errors, th.tensor(delta))
    linear = abs_errors - quadratic
    smooth = 0.5 * quadratic**2 + delta * linear
    return smooth


def vector_projection(u: th.Tensor, v: th.Tensor):
    """projects u onto v"""
    dot_product = th.sum(u * v, dim=1, keepdim=True)
    v_norm_sq = th.sum(v * v, dim=1, keepdim=True)
    # add small epsilon for numerical stability
    projection = (dot_product / (v_norm_sq + 1e-8)) * v
    return projection


def unit_test():
    # test safe_atan2
    numerator = th.tensor([1e-20], requires_grad=True)
    denominator = th.tensor([1e-20], requires_grad=True)
    # out = th.atan2(numerator, denominator) # bad
    out = safe_atan2(numerator, denominator)  # good
    out.mean().backward()
    print(numerator.grad)  # tensor([inf])
    print(denominator.grad)  # tensor([-inf])

    # test safe_normalize
    vec = th.tensor([[0.0, 0.0, 0.0]], requires_grad=True)  # (1, 3)
    vec = th.tensor([[1.0, 2.0, 2.0]], requires_grad=True)  # (1, 3)
    print(vec.shape)
    print(unit_vec)
    unit_vec = F.normalize(vec)
    print(unit_vec)
    output = unit_vec.sum()
    output.backward()
    print(vec.grad)


if __name__ == "__main__":
    unit_test()
