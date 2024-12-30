import torch
def generate_pixel_grid(H, W, device):
    """
    Generate a pixel grid.

    Args:
        H: Height of the image.
        W: Width of the image.
        device: Device to place the tensor on.

    Returns:
        pixel_grid: Tensor of shape (H, W, 2) representing the pixel grid.
    """
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    pixel_grid = torch.stack((x, y), dim=-1)
    return pixel_grid

def pixel_to_camera(pixel_grid, focal, c):
    """
    Convert pixel coordinates to camera coordinates.

    Args:
        pixel_grid: Tensor of shape (H, W, 2) representing the pixel grid.
        pose1: Tensor of shape (ON, X-RN, 4, 4) representing the c2w pose of the first camera.
        focal: Tensor of shape (ON, X-RN,2) representing the focal lengths of the first cameras.
        c: Tensor of shape (ON, X-RN, 2) representing the principal points of  the first cameras.

    Returns:
        camera_coords: Tensor of shape (ON, X-RN, H, W, 3) representing the camera coordinates.
    """
    camera_coords = ((pixel_grid[None, None] - c[:, :, None, None]) / focal[:, :, None, None])
    camera_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)

    return camera_coords


import torch


def world_to_pixel(world_coords, poses_ref, focal_ref, c_ref):
    ON, X_RN, _, _,  _= world_coords.shape
    RN, _, _, _ = poses_ref.shape
    H = 7
    W = 7
    pixels_ref = torch.zeros(ON, X_RN, RN, H, W, 2, device=world_coords.device)

    for i in range(ON):
        for j in range(X_RN):
            for k in range(RN):
                # Construct the camera intrinsic matrix
                K = torch.tensor([[focal_ref[i, k, 0], 0, c_ref[i, k, 0]],
                                  [0, focal_ref[i, k, 1], c_ref[i, k, 1]],
                                  [0, 0, 1]], device=world_coords.device)

                print(world_coords[i, j].shape)
                # Transform world coordinates to camera coordinates
                camera_coords = (torch.matmul(world_coords[i, j], poses_ref[i, k]))
                print(camera_coords.shape)
                # Project camera coordinates to pixel coordinates
                pixels = torch.matmul(K, camera_coords)
                pixels_ref[i, j, k] = pixels.transpose(-1, -2)

    return pixels_ref


def normalize_coordinates(pixels_ref, W, H):
    """
    Normalize pixel coordinates to [-1, 1] range.

    Args:
        pixels_ref: Tensor of shape (ON, N-RN, RN, H, W, 2) representing the pixel coordinates for reference images.
        W: Width of the image.
        H: Height of the image.

    Returns:
        uv_feats: Tensor of shape (ON, N-RN, RN, H, W, 1, 2) representing the normalized pixel coordinates.
    """
    uv_feats = 2 * pixels_ref / torch.tensor([W, H], device=pixels_ref.device).reshape(1, 1, 1, 1, 1, 2) - 1.0
    return uv_feats

def camera_to_world(camera_coords, pose1):
    """
    Convert camera coordinates to world coordinates.

    Args:
        camera_coords: Tensor of shape (ON, H, W, 3) representing the camera coordinates.
        pose1: Tensor of shape (ON, 4, 4) representing the c2w pose of the first camera.

    Returns:
        world_coords: Tensor of shape (ON, H, W, 3) representing the world coordinates.
    """
    # Add singleton dimensions to camera_coords to match the shape of pose1_inv[..., :3, :3].transpose(-1, -2)
    camera_coords = camera_coords.unsqueeze(1)
    print(camera_coords.shape)
    # Transpose pose1 to match the shape of camera_coords for batched matrix multiplication
    pose1 = pose1.transpose(-1, -2)
    print(pose1.shape)
    # Perform matrix multiplication
    world_coords = torch.matmul(camera_coords, pose1[..., :3, :3]) + pose1[..., :3, 3:4]
    return world_coords

def transform_images_to_reference_image_coordinates(pose1, poses_ref, focal, c, focal_ref, c_ref):
    """
    Transform points from the coordinate system of image1 to the coordinate system of reference images.

    Args:
        pose1: Tensor of shape (ON, 4, 4) representing the c2w pose of the first camera.
        poses_ref: Tensor of shape (ON, RN, 4, 4) representing the w2c poses of reference cameras.
        focal: Tensor of shape (ON, 2) representing the focal lengths of the first cameras.
        c: Tensor of shape (ON, 2) representing the principal points of the first cameras.
        focal_ref: Tensor of shape (ON, 2) representing the focal lengths of reference cameras.
        c_ref: Tensor of shape (ON, 2) representing the principal points of reference cameras.

    Returns:
        uv_feats: Transformed points in the coordinate system of reference images, compatible with F.grid_sample.
    """
    ON, X_RN, _, _ = pose1.shape
    _, RN, _, _ = poses_ref.shape
    H, W = 7, 7  # Assuming fixed size for simplicity, adjust according to your actual data
    # Generate pixel grid for image1
    focal=focal.unsqueeze(1).expand(ON, X_RN, 2)  # (ON, X_RN, 2)
    c=c.unsqueeze(1).expand(ON,X_RN,2)
    focal_ref=focal_ref.unsqueeze(1).expand(ON, RN, 2)  # (ON, RN, 2)
    c_ref=c_ref.unsqueeze(1).expand(ON,RN,2)
    pixel_grid1 = generate_pixel_grid(H, W, device=pose1.device)
    pixel_grid1 = pixel_grid1.to(focal.device)
    # Convert pixel grid to camera coordinates for image1
    camera_coords1 = pixel_to_camera(pixel_grid1, focal, c)
    # Convert camera coordinates to world coordinates for image1
    world_coords1 = camera_to_world(camera_coords1, pose1)
    # Convert world coordinates to pixel coordinates for reference images
    pixels_ref = world_to_pixel(world_coords1, poses_ref, focal_ref, c_ref)
    # Normalize pixel coordinates to [-1, 1] range
    uv_feats = normalize_coordinates(pixels_ref, W, H)
    return uv_feats

# Define other helper functions (generate_pixel_grid, pixel_to_camera, world_to_pixel, normalize_coordinates) as before

# Example usage
#pose1 = torch.randn(2, 2, 4, 4)  # Example c2w pose of the first camera
#poses_ref = torch.randn(2, 3, 4, 4)  # Example w2c poses of reference cameras
#focal = torch.randn(2, 2)  # Example focal lengths of the first cameras
#c = torch.randn(2, 2)  # Example principal points of the first cameras
#focal_ref = torch.randn(2, 2)  # Example focal lengths of reference cameras
#_ref = torch.randn(2, 2)  # Example principal points of reference cameras

# Transform points to reference image coordinates
#uv_feats = transform_images_to_reference_image_coordinates(pose1, poses_ref, focal, c, focal_ref, c_ref)

def camera_to_world1(camera_coords, c2w_pose):
    print(camera_coords.shape)
    # 将相机坐标扩展为齐次坐标形式
    homogeneous_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)
    print(homogeneous_coords.shape)
    print(c2w_pose.shape)
    #b=homogeneous_coords.unsqueeze(-1)
    #print(b.shape)
    # 将相机坐标系上的点转换到世界坐标系上
    world_coords = torch.matmul(homogeneous_coords,c2w_pose)
    print(world_coords.shape)
    # 去除齐次坐标，保留前三个坐标
    world_coords = world_coords[..., :3]
    print(world_coords.shape)
    return world_coords
"""

# Example usage
ON = 2
X_RN = 175
RN = 3
H = 7
W = 7
world_coords = torch.randn(ON, X_RN, H, W, 4)  # Example world coordinates
poses_ref = torch.randn(ON, RN, 4, 4)  # Example reference camera poses
focal_ref = torch.randn(ON, RN, 2)  # Example focal lengths of reference cameras
c_ref = torch.randn(ON, RN, 2)  # Example principal points of reference cameras

# Convert world coordinates to pixel coordinates
pixels_ref = world_to_pixel(world_coords, poses_ref, focal_ref, c_ref)

print("Pixel coordinates shape:", pixels_ref.shape)
"""
#a = torch.randn(3,3)
#b = torch.randn(49,3)
#c= torch.matmul(a,b.transpose(-1, -2))
#print(c.shape)

torch.sin(torch.addcmul(self._phases, embed, self._freqs))