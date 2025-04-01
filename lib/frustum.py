import torch


def shuffle_uv(focals: torch.Tensor, width: int, height: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    """
    Shuffle the UV coordinates of the image plane.

    Args:
    - focals: torch.Tensor of shape (N)
    - width: int
    - height: int
    - randomize: bool

    Returns:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - u: torch.Tensor of shape (H, W)
    - v: torch.Tensor of shape (H, W)
    """
    focals = focals.to(device)
    u, v = torch.meshgrid(torch.linspace(0, width - 1, width, device=device, dtype=dtype), torch.linspace(0, height - 1, height, device=device, dtype=dtype), indexing='xy')  # (H, W), (H, W)
    if randomize:
        du, dv = torch.rand_like(u), torch.rand_like(v)  # (H, W), (H, W)
        u, v = torch.clip(u + du, 0, width - 1), torch.clip(v + dv, 0, height - 1)  # (H, W), (H, W)
    u_normalized, v_normalized = (u - width * 0.5) / focals[:, None, None], (v - height * 0.5) / focals[:, None, None]  # (N, H, W), (N, H, W)
    dirs = torch.stack([u_normalized, -v_normalized, -torch.ones_like(u_normalized)], dim=-1)  # (N, H, W, 3)
    dirs_normalized = torch.nn.functional.normalize(dirs, dim=-1)

    return dirs_normalized, u, v


def resample_frames(frames: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """
    Resample frames using the given UV coordinates.

    Args:
    - frames: torch.Tensor of shape (..., H, W, C)
    - u: torch.Tensor of shape (N, H, W)
    - v: torch.Tensor of shape (N, H, W)

    Returns:
    - resampled_images: torch.Tensor of shape (N, T, H, W, C)
    """

    H, W, C = frames.shape[-3:]
    u_norm, v_norm = 2.0 * (u / (W - 1)) - 1, 2.0 * (v / (H - 1)) - 1
    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(0)  # (1, H, W, 2)
    orig_shape = frames.shape
    reshaped_images = frames.reshape(-1, H, W, C).permute(0, 3, 1, 2)  # (batch, C, H, W)
    resampled = torch.nn.functional.grid_sample(reshaped_images, grid.expand(reshaped_images.shape[0], -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)
    resampled_images = resampled.permute(0, 2, 3, 1).reshape(orig_shape)
    return resampled_images


def sample_frustum(dirs: torch.Tensor, poses: torch.Tensor, batch_size: int, randomize: bool, device: torch.device):
    """
    Sample points in the frustum of each camera.

    Args:
    - dirs: torch.Tensor of shape (N, H, W, 3)
    - poses: torch.Tensor of shape (N, 4, 4)
    - near: float
    - far: float
    - depth: int
    - batch_size: int
    - randomize: bool

    Yields:
    - points: torch.Tensor of shape (batch_size, depth, 3)
    """

    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (N, H, W, 3)

    rays_d = rays_d.reshape(-1, 3)  # (N*H*W, 3)
    rays_o = rays_o.reshape(-1, 3)  # (N*H*W, 3)
    num_rays = rays_d.shape[0]

    if randomize:
        indices = torch.randperm(num_rays, device=device)  # (N*H*W)
    else:
        indices = torch.arange(num_rays, device=device)  # (N*H*W)

    for i in range(0, num_rays, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_rays_o = rays_o[batch_indices]  # (batch_size, 3)
        batch_rays_d = rays_d[batch_indices]  # (batch_size, 3)
        yield batch_indices, batch_rays_o, batch_rays_d


def sample_random_frame(videos_data: torch.Tensor, batch_indices: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    Sample a random frame from the given videos data.

    Args:
    - videos_data: torch.Tensor of shape (T, V, H, W, C)
    - batch_indices: torch.Tensor of shape (batch_size)

    Returns:
    - batch_time: torch.Tensor of shape (1)
    - batch_target_pixels: torch.Tensor of shape (batch_size, C)
    """
    frame = torch.rand((), device=device, dtype=dtype) * (videos_data.shape[0] - 1)
    frame_floor = torch.floor(frame).long()
    frame_ceil = frame_floor + 1
    frames_alpha = frame - frame_floor.to(frame.dtype)
    target_frame = (1 - frames_alpha) * videos_data[frame_floor] + frames_alpha * videos_data[frame_ceil]  # (V * H * W, C)
    target_frame = target_frame.reshape(-1, 3)
    batch_target_pixels = target_frame[batch_indices]  # (batch_size, C)
    batch_time = frame / (videos_data.shape[0] - 1)

    return batch_time, batch_target_pixels


def refresh_generator(batch_size, videos_data, poses, focals, width, height, target_device, target_dtype):
    dirs, u, v = shuffle_uv(focals=focals, width=width, height=height, randomize=True, device=torch.device("cpu"), dtype=target_dtype)
    dirs = dirs.to(target_device)
    videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)
    generator = sample_frustum(dirs=dirs, poses=poses, batch_size=batch_size, randomize=True, device=target_device)

    return generator, videos_data_resampled


def sample_bbox(resx, resy, resz, batch_size, randomize, target_device, target_dtype, s2w, s_w2s, s_scale, s_min, s_max):
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=target_device, dtype=target_dtype), torch.linspace(0, 1, resy, device=target_device, dtype=target_dtype), torch.linspace(0, 1, resz, device=target_device, dtype=target_dtype)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
    coord_3d_world = sim2world(coord_3d_sim, s2w, s_scale)
    bbox_mask = insideMask(coord_3d_world, s_w2s, s_scale, s_min, s_max, to_float=False)
    coord_3d_world_filtered = coord_3d_world[bbox_mask].reshape(-1, 3)

    num_points = coord_3d_world_filtered.shape[0]

    if randomize:
        indices = torch.randperm(num_points, device=target_device)
    else:
        indices = torch.arange(num_points, device=target_device)

    for i in range(0, num_points, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_points = coord_3d_world_filtered[batch_indices]
        yield batch_points


def get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)
    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)

        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac


def pos_world2smoke(inputs_pts, s_w2s, s_scale):
    pts_world_homo = torch.cat([inputs_pts, torch.ones_like(inputs_pts[..., :1])], dim=-1)
    pts_sim_ = torch.matmul(s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
    pts_sim = pts_sim_ / s_scale
    return pts_sim


def isInside(inputs_pts, s_w2s, s_scale, s_min, s_max):
    target_pts = pos_world2smoke(inputs_pts, s_w2s, s_scale)
    above = torch.logical_and(target_pts[..., 0] >= s_min[0], target_pts[..., 1] >= s_min[1])
    above = torch.logical_and(above, target_pts[..., 2] >= s_min[2])
    below = torch.logical_and(target_pts[..., 0] <= s_max[0], target_pts[..., 1] <= s_max[1])
    below = torch.logical_and(below, target_pts[..., 2] <= s_max[2])
    outputs = torch.logical_and(below, above)
    return outputs


def insideMask(inputs_pts, s_w2s, s_scale, s_min, s_max, to_float=False):
    mask = isInside(inputs_pts, s_w2s, s_scale, s_min, s_max)
    return mask.to(torch.float) if to_float else mask


def sim2world(pts_sim, s2w, s_scale):
    pts_sim_ = pts_sim * s_scale
    pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
    pts_world = torch.matmul(s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
    return pts_world


def world2sim(pts_world, s_w2s, s_scale):
    pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
    pts_sim_ = torch.matmul(s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
    pts_sim = pts_sim_ / s_scale  # 3.target to 2.simulation
    return pts_sim


def world2sim_rot(pts_world, s_w2s, s_scale):
    pts_sim_ = torch.matmul(s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
    pts_sim = pts_sim_ / (s_scale)  # 3.target to 2.simulation
    return pts_sim
