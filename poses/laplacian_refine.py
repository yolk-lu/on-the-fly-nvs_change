import torch
import torch.nn.functional as F

def get_laplacian_edges(img_tensor):
    """ 
    Extract structural edges using the mathematical Laplacian Operator 
    Expects img_tensor to be shape [B, 3, H, W].
    """
    device = img_tensor.device
    laplacian_kernel = torch.tensor(
        [[[0,  1,  0], 
          [1, -4,  1], 
          [0,  1,  0]]], dtype=torch.float32, device=device
    ).unsqueeze(0).repeat(3, 1, 1, 1)
    
    laplacian = F.conv2d(img_tensor, laplacian_kernel, padding=1, groups=3)
    # Get magnitude of edges across color channels
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=2, dim=1, keepdim=True)
    
    # Normalize purely for visual structural thresholding
    laplacian_norm = laplacian_norm / laplacian_norm.max().clamp_min(1e-5)
    return laplacian_norm


def refine_depth_with_laplacian(img_tensor, depth_tensor, iters=200, lambda_smooth=5.0):
    """
    Uses the Laplacian Operator's boundary information to pull fractured depth values
    into alignment with actual structural corners.
    
    Args:
        img_tensor: [B, 3, H, W] RGB image normalized to [0, 1]
        depth_tensor: [B, 1, H, W] Neural predicted depth map
        iters: Overfitting optimization loop counts
        lambda_smooth: Factor enforcing aggressive cliff drops at RGB boundaries
        
    Returns:
        edges: [B, 1, H, W] Tensor representing true RGB boundaries
        D_refined: [B, 1, H, W] The boundary-corrected depth tensor
    """
    edges = get_laplacian_edges(img_tensor)
    
    # Edge confidence mask: 1.0 means strong RGB boundary, 0.0 means flat surface.
    # Inverse to create a smoothness constraint: We want high smoothness where there are NO edges!
    smoothness_mask = torch.exp(-15.0 * edges) 
    
    # 2D Laplacian spatial constraint operator for Depth Map
    device = depth_tensor.device
    depth_lap_kernel = torch.tensor(
        [[[[0,  1,  0], 
           [1, -4,  1], 
           [0,  1,  0]]]], dtype=torch.float32, device=device
    )
    
    # Optimize a new refined depth map guided purely by the Laplacian masks
    # D_refined starts identical to network prediction
    D_refined = depth_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([D_refined], lr=0.01)
    
    for _ in range(iters):
        optimizer.zero_grad()
        
        # 1. Base Fidelity: Keep it grounded to the Neural Network's macro prediction
        loss_data = F.mse_loss(D_refined, depth_tensor)
        
        # 2. Laplacian Completion / Extrapolation:
        # Penalize any depth shifts (Laplacian != 0) inside solid objects,
        # but ALLOW depth shifts to exist cleanly across RGB boundaries (smoothness_mask is 0 there).
        # This mathematically forces broken/bled depth edges to snap sharply onto building corners.
        D_lap = F.conv2d(D_refined, depth_lap_kernel, padding=1)
        loss_smooth = (D_lap.abs() * smoothness_mask).mean()
        
        loss = loss_data + lambda_smooth * loss_smooth
        loss.backward()
        optimizer.step()
        
    return edges, D_refined.detach()
