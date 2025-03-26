import torch

from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *

import math
import tqdm

training_videos_scene1_rotating = [
    "data/PISG/scene1/rotating.mp4",
]

camera_calibrations_scene1_rotating = [
    "data/PISG/scene1/cam_rotating.npz",
]


def sample_frustum_rotating(dirs: torch.Tensor, poses: torch.Tensor, batch_size: int, randomize: bool, device: torch.device, dtype: torch.dtype):
    rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (N, H, W, 3)
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (N, H, W, 3)
    N, H, W = rays_d.shape[:3]

    offsets = torch.zeros(N, device=device)

    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)

    indices = torch.arange(N * H * W, device=device).view(N, -1)
    if randomize:
        for i in range(N):
            indices[i] = indices[i][torch.randperm(H * W, device=device)]
    current_pos = torch.zeros(N, device=device, dtype=torch.long)

    while (current_pos < H * W).any():
        per_sample = batch_size // N
        remainder = batch_size % N

        batch_indices_list = []
        batch_frame_float_list = []
        for i in range(N):
            k = per_sample + (1 if i < remainder else 0)
            start = current_pos[i].item()
            end = min(start + k, H * W)

            if start < end:
                selected = indices[i, start:end]
                batch_indices_list.append(selected)

                # 插值后的实际帧位置：i + offset[i]
                frame_f = i + offsets[i].item()
                batch_frame_float_list.append(torch.full((end - start,), frame_f, device=device, dtype=dtype))

                current_pos[i] += end - start

        if batch_indices_list:
            batch_indices = torch.cat(batch_indices_list, dim=0)       # (B,)
            batch_frame = torch.cat(batch_frame_float_list, dim=0)     # (B,)
            batch_rays_o = rays_o_flat[batch_indices]                  # (B, 3)
            batch_rays_d = rays_d_flat[batch_indices]                  # (B, 3)

            yield batch_indices, batch_frame, batch_rays_o, batch_rays_d

@torch.compile
def volume_render(batch_indices, batch_frame, batch_rays_o, batch_rays_d, videos_data_resampled_flat, depth_size, target_device, target_dtype, NEAR_float, FAR_float, encoder_d, model_d):
    batch_target_pixels = videos_data_resampled_flat[batch_indices]
    batch_time = batch_frame / (120 - 1)

    batch_size_current = batch_rays_d.shape[0]

    t_vals = torch.linspace(0., 1., steps=depth_size, device=target_device, dtype=target_dtype)
    t_vals = t_vals.view(1, depth_size)
    z_vals = NEAR_float * (1. - t_vals) + FAR_float * t_vals
    z_vals = z_vals.expand(batch_size_current, depth_size)

    mid_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], -1)
    lower_vals = torch.cat([z_vals[..., :1], mid_vals], -1)
    t_rand = torch.rand(z_vals.shape, device=target_device, dtype=target_dtype)
    z_vals = lower_vals + (upper_vals - lower_vals) * t_rand

    batch_dist_vals = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size_current, N_depths-1]
    batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * z_vals[..., :, None]  # [batch_size_current, N_depths, 3]
    batch_points_time = torch.cat([batch_points, batch_time.view(-1, 1, 1).expand(batch_points[..., :1].shape)], dim=-1)
    batch_points_time_flat = batch_points_time.reshape(-1, 4)

    batch_points_time_flat_filtered = batch_points_time_flat

    hidden = encoder_d(batch_points_time_flat_filtered)
    raw_flat = model_d(hidden)
    raw = raw_flat.reshape(batch_size_current, depth_size, 1)

    dists_cat = torch.cat([batch_dist_vals, torch.tensor([1e10], device=target_device).expand(batch_dist_vals[..., :1].shape)], -1)  # [batch_size_current, N_depths]
    dists_final = dists_cat * torch.norm(batch_rays_d[..., None, :], dim=-1)  # [batch_size_current, N_depths]

    rgb_trained = torch.ones(3, device=target_device) * (0.6 + torch.tanh(model_d.rgb) * 0.4)
    noise = 0.
    alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1] + noise) * dists_final)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=target_device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)
    img_loss = torch.nn.functional.mse_loss(rgb_map, batch_target_pixels)

    return img_loss

if __name__ == '__main__':
    ratio = 0.5
    target_device = torch.device("cuda:0")
    target_dtype = torch.float32
    videos_data = load_videos_data(*training_videos_scene1_rotating, ratio=ratio, dtype=target_dtype)
    poses, focals, widths, heights, nears, fars = load_rotating_camera_data(*camera_calibrations_scene1_rotating, ratio=ratio, device=target_device, dtype=target_dtype)

    encoder_d = HashEncoderNativeFasterBackward().to(target_device)
    model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_d.num_levels * 2).to(target_device)
    optimizer_d = torch.optim.RAdam([{'params': model_d.parameters(), 'weight_decay': 1e-6}, {'params': encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

    target_lr_ratio = 0.1
    gamma = math.exp(math.log(target_lr_ratio) / 30000)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=gamma)

    dirs, u, v = shuffle_uv(focals=focals[0], width=int(widths[0, 0].item()), height=int(heights[0, 0].item()), randomize=True, device=torch.device("cpu"), dtype=target_dtype)
    dirs = dirs.to(target_device)
    videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)
    videos_data_resampled_flat = videos_data_resampled.reshape(-1, 3)

    batch_size = 1024
    depth_size = 192
    global_step = 1
    NEAR_float = float(nears[0, 0].item())
    FAR_float = float(fars[0, 0].item())
    generator = sample_frustum_rotating(dirs=dirs, poses=poses[0], batch_size=batch_size, randomize=True, device=target_device, dtype=target_dtype)

    def save_ckpt(directory: str):
        from datetime import datetime
        timestamp = datetime.now().strftime('%m%d%H')
        filename = 'ckpt_{}_bs{}_{:06d}.tar'.format(timestamp, batch_size, global_step)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save({
            'global_step': global_step,
            'model_d': model_d.state_dict(),
            'encoder_d': encoder_d.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, path)

    for _ in tqdm.trange(2000):
        try:
            batch_indices, batch_frame, batch_rays_o, batch_rays_d = next(generator)
            img_loss = volume_render(batch_indices, batch_frame, batch_rays_o, batch_rays_d, videos_data_resampled_flat, depth_size, target_device, target_dtype, NEAR_float, FAR_float, encoder_d, model_d)
            img_loss.backward()
            optimizer_d.step()
            global_step += 1
            tqdm.tqdm.write(f"iter: {global_step}, lr_d: {scheduler_d.get_last_lr()[0]}, img_loss: {img_loss}")

        except StopIteration:
            print("No more data to sample.")
            break
    save_ckpt("ckpt")
