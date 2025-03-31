import torch
from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *

import math
import tqdm
import os


def refresh_generator(batch_size, videos_data, poses, focals, width, height, target_device, target_dtype):
    dirs, u, v = shuffle_uv(focals=focals, width=width, height=height, randomize=True, device=torch.device("cpu"), dtype=target_dtype)
    dirs = dirs.to(target_device)
    videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)
    generator = sample_frustum(dirs=dirs, poses=poses, batch_size=batch_size, randomize=True, device=target_device)

    return generator, videos_data_resampled


class TrainModel:
    def __init__(self, batch_size: int, ratio: float, target_device: torch.device, target_dtype: torch.dtype):
        self._load_model(target_device)
        self._load_dataset(ratio, target_device, target_dtype)
        self._load_valid_domain(target_device, target_dtype)
        self.target_device = target_device
        self.target_dtype = target_dtype
        self.batch_size = batch_size

        self.generator, self.videos_data_resampled = refresh_generator(self.batch_size, self.videos_data, self.poses, self.focals, int(self.width[0].item()), int(self.height[0].item()), self.target_device, self.target_dtype)
        self.global_step = 0

    def _load_model(self, target_device: torch.device):
        self.encoder_d = HashEncoderNativeFasterBackward().to(target_device)
        self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        self.optimizer_d = torch.optim.RAdam([{'params': self.model_d.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
        self.encoder_v = HashEncoderNativeFasterBackward().to(target_device)
        self.model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_v.num_levels * 2, use_f=False).to(target_device)
        self.optimizer_v = torch.optim.RAdam([{'params': self.model_v.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder_v.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        target_lr_ratio = 0.1
        gamma = math.exp(math.log(target_lr_ratio) / 30000)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=gamma)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, gamma=gamma)

    def _load_dataset(self, ratio: float, target_device: torch.device, target_dtype: torch.dtype):
        self.videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
        self.poses, self.focals, self.width, self.height, self.near, self.far = load_cameras_data(*camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)

    def _load_valid_domain(self, target_device: torch.device, target_dtype: torch.dtype):
        VOXEL_TRAN = torch.tensor([
            [1.0, 0.0, 7.5497901e-08, 8.1816666e-02],
            [0.0, 1.0, 0.0, -4.4627272e-02],
            [7.5497901e-08, 0.0, -1.0, -4.9089999e-03],
            [0.0, 0.0, 0.0, 1.0]
        ], device=target_device, dtype=target_dtype)
        VOXEL_SCALE = torch.tensor([0.4909, 0.73635, 0.4909], device=target_device, dtype=target_dtype)
        self.s_w2s = torch.inverse(VOXEL_TRAN).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = VOXEL_SCALE.expand([3])
        self.s_min = torch.tensor([0.15, 0.0, 0.15], device=target_device, dtype=target_dtype)
        self.s_max = torch.tensor([0.85, 1.0, 0.85], device=target_device, dtype=target_dtype)

    def optimize_density(self, depth_size: int):
        self.optimizer_d.zero_grad()
        try:
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        except StopIteration:
            self.generator, self.videos_data_resampled = refresh_generator(self.batch_size, self.videos_data, self.poses, self.focals, int(self.width[0].item()), int(self.height[0].item()), self.target_device, self.target_dtype)
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        img_loss = self.image_loss(batch_indices, batch_rays_o, batch_rays_d, depth_size, float(self.near[0].item()), float(self.far[0].item()))
        img_loss.backward()
        self.optimizer_d.step()
        self.scheduler_d.step()
        self.global_step += 1
        tqdm.tqdm.write(f"iter: {self.global_step}, lr_d: {self.scheduler_d.get_last_lr()[0]}, img_loss: {img_loss}")

    def image_loss(self, batch_indices, batch_rays_o, batch_rays_d, depth_size: int, near: float, far: float):
        batch_time, batch_target_pixels = sample_random_frame(videos_data=self.videos_data_resampled, batch_indices=batch_indices, device=self.target_device, dtype=self.target_dtype)
        batch_size_current = batch_rays_d.shape[0]

        t_vals = torch.linspace(0., 1., steps=depth_size, device=self.target_device, dtype=self.target_dtype)
        t_vals = t_vals.view(1, depth_size)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand(batch_size_current, depth_size)

        mid_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper_vals = torch.cat([mid_vals, z_vals[..., -1:]], -1)
        lower_vals = torch.cat([z_vals[..., :1], mid_vals], -1)
        t_rand = torch.rand(z_vals.shape, device=self.target_device, dtype=self.target_dtype)
        z_vals = lower_vals + (upper_vals - lower_vals) * t_rand

        batch_dist_vals = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size_current, N_depths-1]
        batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * z_vals[..., :, None]  # [batch_size_current, N_depths, 3]
        batch_points_time = torch.cat([batch_points, batch_time.expand(batch_points[..., :1].shape)], dim=-1)
        batch_points_time_flat = batch_points_time.reshape(-1, 4)

        bbox_mask = self.points_mask(batch_points_time_flat[..., :3])
        batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]

        hidden = self.encoder_d(batch_points_time_flat_filtered)
        raw_d = self.model_d(hidden)
        raw_flat = torch.zeros([batch_size_current * depth_size], device=self.target_device, dtype=self.target_dtype)
        raw_d_flat = raw_d.view(-1)
        raw_flat = raw_flat.masked_scatter(bbox_mask, raw_d_flat)
        raw = raw_flat.reshape(batch_size_current, depth_size, 1)

        dists_cat = torch.cat([batch_dist_vals, torch.tensor([1e10], device=self.target_device).expand(batch_dist_vals[..., :1].shape)], -1)  # [batch_size_current, N_depths]
        dists_final = dists_cat * torch.norm(batch_rays_d[..., None, :], dim=-1)  # [batch_size_current, N_depths]

        rgb_trained = torch.ones(3, device=self.target_device) * (0.6 + torch.tanh(self.model_d.rgb) * 0.4)
        noise = 0.
        alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1] + noise) * dists_final)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.target_device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)
        img_loss = torch.nn.functional.mse_loss(rgb_map, batch_target_pixels)

        return img_loss

    def points_mask(self, batch_points):
        return insideMask(batch_points, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)

    def save_ckpt(self, directory: str):
        from datetime import datetime
        timestamp = datetime.now().strftime('%m%d%H')
        filename = 'ckpt_{}_bs{}_{:06d}.tar'.format(timestamp, self.batch_size, self.global_step)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        torch.save({
            'global_step': self.global_step,
            'model_d': self.model_d.state_dict(),
            'encoder_d': self.encoder_d.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'model_v': self.model_v.state_dict(),
            'encoder_v': self.encoder_v.state_dict(),
            'optimizer_v': self.optimizer_v.state_dict(),
        }, path)

    def load_ckpt(self, path: str):
        checkpoint = torch.load(path)
        self.global_step = checkpoint['global_step']
        self.model_d.load_state_dict(checkpoint['model_d'])
        self.encoder_d.load_state_dict(checkpoint['encoder_d'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.model_v.load_state_dict(checkpoint['model_v'])
        self.encoder_v.load_state_dict(checkpoint['encoder_v'])
        self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])
        tqdm.tqdm.write(f"loaded checkpoint from {path}")


def train_density_only(total_iter, batch_size, depth_size, ratio, target_device, target_dtype, pretrained_ckpt=None):
    model = TrainModel(batch_size, ratio, target_device, target_dtype)
    if pretrained_ckpt:
        model.load_ckpt(pretrained_ckpt)
    try:
        for _ in tqdm.trange(total_iter):
            model.optimize_density(depth_size)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt('houdini/ckpt_joint')


if __name__ == "__main__":
    # option1 - train_density_only
    train_density_only(
        total_iter=1000,
        batch_size=1024,
        depth_size=192,
        ratio=0.5,
        target_device=torch.device("cuda:0"),
        target_dtype=torch.float32,
    )
