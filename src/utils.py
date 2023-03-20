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


# Converts NED poses to standard axes
# @input ned [torch.tensor (B,6)]: poses in NED with ZYX Euler angles (x, y, z, alpha, beta, gamma)
# @output [torch.tensor (B,6)]: points in NWU axes with ZYX euler angles
def ned_to_nwu(ned: torch.tensor) -> torch.tensor:
    return torch.stack((ned[:,0], -ned[:,1], -ned[:,2],
                        -ned[:,3], -ned[:,4], ned[:,5]), dim=1)


# Converts standard poses to NED axes
# @input nwu [torch.tensor (B,6)]: poses in NWU axes with ZYX Euler angles (x, y, z, alpha, beta, gamma)
# @output [torch.tensor (B,6)]: points in NED axes with ZYX euler angles
def nwu_to_ned(nwu: torch.tensor) -> torch.tensor:
    return torch.stack((nwu[:,0], -nwu[:,1], -nwu[:,2],
                        -nwu[:,3], -nwu[:,4], nwu[:,5]), dim=1)