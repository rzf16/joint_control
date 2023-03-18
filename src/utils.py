'''
Handy utility functions
'''
import torch


# Wraps angles (in radians) to (-pi, pi]
# @input theta [torch.tensor (B)]: angles in radians
# @output [torch.tensor (B)]: wrapped angles
def wrap_radians(theta: torch.tensor) -> torch.tensor:
    wrapped = torch.remainder(theta, 2*torch.pi)
    wrapped[wrapped > torch.pi] -= 2*torch.pi
    return wrapped