
import torch
import torch.nn.functional as F

# SH Constants
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def eval_sh_basis(deg, dirs):
    """
    Evaluate SH basis functions up to degree 3.
    Args:
        deg (int): Maximum SH degree (currently supports 3).
        dirs (torch.Tensor): Unit vectors of shape (..., 3).
    Returns:
        sh_basis (torch.Tensor): SH basis values of shape (..., (deg+1)^2).
    """
    assert deg <= 3, "Only degree <= 3 is supported."
    
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    
    sh = []
    
    # Band 0
    sh.append(torch.full_like(x, C0)) # Y00
    
    if deg >= 1:
        # Band 1
        sh.append(-C1 * y) # Y1-1
        sh.append( C1 * z) # Y10
        sh.append(-C1 * x) # Y11
        
    if deg >= 2:
        # Band 2
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        
        sh.append(C2[0] * xy)             # Y2-2
        sh.append(C2[1] * yz)             # Y2-1
        sh.append(C2[2] * (2.0 * zz - xx - yy)) # Y20
        sh.append(C2[3] * xz)             # Y21
        sh.append(C2[4] * (xx - yy))      # Y22
        
    if deg >= 3:
        # Band 3
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        
        sh.append(C3[0] * y * (3.0 * xx - yy))         # Y3-3
        sh.append(C3[1] * xy * z)                      # Y3-2
        sh.append(C3[2] * y * (4.0 * zz - xx - yy))    # Y3-1
        sh.append(C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) # Y30
        sh.append(C3[4] * x * (4.0 * zz - xx - yy))    # Y31
        sh.append(C3[5] * z * (xx - yy))               # Y32
        sh.append(C3[6] * x * (xx - 3.0 * yy))         # Y33
        
    return torch.stack(sh, dim=-1) # (..., K)

def compute_sh_from_sg(amplitude, axis, sharpness, degree=3):
    """
    Project Spherical Gaussians (SG) to Spherical Harmonics (SH).
    
    Formula: c_lm = sum_k [ mu_k * A_l(lambda_k) * Y_lm(xi_k) ]
    where A_l(lambda) = exp( -l(l+1) / (2*lambda) )
    
    Args:
        amplitude (torch.Tensor): (N, Lobes, 3) - RGB amplitudes.
        axis (torch.Tensor): (N, Lobes, 3) - Unit axis directions.
        sharpness (torch.Tensor): (N, Lobes, 1) - Sharpness coefficients (lambda).
        degree (int): SH degree (default 3).
        
    Returns:
        f_rest (torch.Tensor): (N, (deg+1)^2 - 1, 3) - SH coefficients (excluding DC).
        f_dc (torch.Tensor): (N, 1, 3) - DC coefficient.
    """
    # 1. Normalize axes
    axis = F.normalize(axis, dim=-1)
    
    # 2. Compute SH Basis at axes: (N, Lobes, K)
    Y_lm = eval_sh_basis(degree, axis)
    
    # 3. Compute Spectral Conv Factors A_l(lambda): (N, Lobes, K)
    # K = (deg+1)^2
    # Determine 'l' for each basis function index
    l_list = []
    for l in range(degree + 1):
        l_list.extend([l] * (2*l + 1))
    l_tensor = torch.tensor(l_list, device=amplitude.device).float() # (K,)
    
    # A_l = exp( - l(l+1) / (2 * sharpness) )
    exponent = - l_tensor * (l_tensor + 1.0) / (2.0 * sharpness) # (N, Lobes, K)
    A_l = torch.exp(exponent)
    
    # 4. Integrate contributions
    # Contribution of lobe k to coeff i: mu_k[RGB] * A_l[i] * Y_lm[i]
    # Shape: (N, Lobes, 3) * (N, Lobes, K) -> (N, Lobes, K, 3)
    
    # Make dims compatible
    # amp: (N, L, 3) -> (N, L, 1, 3)
    # A_l: (N, L, K) -> (N, L, K, 1)
    # Y_lm: (N, L, K) -> (N, L, K, 1)
    
    contributions = amplitude.unsqueeze(2) * (A_l * Y_lm).unsqueeze(3) # (N, L, K, 3)
    
    # Sum over Lobes dim=1 -> (N, K, 3)
    sh_coeffs = contributions.sum(dim=1) 
    
    # Split DC and Rest
    f_dc = sh_coeffs[:, 0:1, :] # (N, 1, 3)
    f_rest = sh_coeffs[:, 1:, :] # (N, 15, 3)
    
    return f_dc, f_rest
