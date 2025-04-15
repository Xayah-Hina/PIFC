from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *

import torch
import dataclasses


@dataclasses.dataclass
class TrainConfig:
    # datasets
    scene_name: str

    # general parameters
    target_device: torch.device
    target_dtype: torch.dtype

    # training parameters
    batch_size: int
    depth_size: int
    ratio: float

    # ckpt parameters
    use_mid_ckpts: bool
    mid_ckpts_iters: int

    def __post_init__(self):
        import yaml
        import os
        scene_info_path = f'data/{self.scene_name}/scene_info.yaml'
        assert os.path.exists(scene_info_path), f"Scene info file not found: {scene_info_path}"
        with open(scene_info_path, 'r') as f:
            scene_info = yaml.safe_load(f)
            self.training_videos = scene_info['training_videos']
            self.camera_calibrations = scene_info['camera_calibrations']
            self.voxel_transform = torch.tensor(scene_info['voxel_transform'], device=self.target_device, dtype=self.target_dtype)
            self.voxel_scale = torch.tensor(scene_info['voxel_scale'], device=self.target_device, dtype=self.target_dtype)
            self.s_min = torch.tensor(scene_info['s_min'], device=self.target_device, dtype=self.target_dtype)
            self.s_max = torch.tensor(scene_info['s_max'], device=self.target_device, dtype=self.target_dtype)
        self.s_w2s = torch.inverse(self.voxel_transform).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = self.voxel_scale.expand([3])


class _TrainModelBase:
    def __init__(self, config: TrainConfig):
        self._reinitialize(config)

    def _reinitialize(self, config):
        self._load_model(config.target_device)
        self._load_dataset(config.training_videos, config.camera_calibrations, config.ratio, config.target_device, config.target_dtype)

        self.target_device = config.target_device
        self.target_dtype = config.target_dtype
        self.scene_name = config.scene_name
        self.s_w2s, self.s2w, self.s_scale, self.s_min, self.s_max = config.s_w2s, config.s2w, config.s_scale, config.s_min, config.s_max

        self.generator = None
        self.videos_data_resampled = None
        self.global_step = 0

        self.config = config  # Don't use is unless save_ckpt

    def _load_model(self, target_device: torch.device):
        self.encoder_d = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        self.optimizer_d = torch.optim.RAdam([{'params': self.model_d.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
        self.encoder_v = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_v.num_levels * 2, use_f=False).to(target_device)
        self.optimizer_v = torch.optim.RAdam([{'params': self.model_v.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder_v.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        target_lr_ratio = 0.001
        import math
        gamma = math.exp(math.log(target_lr_ratio) / 100000)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=gamma)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, gamma=gamma)

    def _load_dataset(self, training_videos, camera_calibrations, ratio: float, target_device: torch.device, target_dtype: torch.dtype):
        assert len(training_videos) == len(camera_calibrations), "Number of videos and camera calibrations must match."
        self.videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
        self.poses, self.focals, self.width, self.height, self.near, self.far = load_cameras_data(*camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)

    def _next_batch(self, batch_size: int):
        if self.generator is None:
            self.generator, self.videos_data_resampled = refresh_generator(batch_size, self.videos_data, self.poses, self.focals, int(self.width[0].item()), int(self.height[0].item()), self.target_device, self.target_dtype)
        try:
            return next(self.generator)
        except StopIteration:
            self.generator, self.videos_data_resampled = refresh_generator(batch_size, self.videos_data, self.poses, self.focals, int(self.width[0].item()), int(self.height[0].item()), self.target_device, self.target_dtype)
            return next(self.generator)

    def save_ckpt(self, directory: str, final: bool):
        from datetime import datetime
        timestamp = datetime.now().strftime('%m%d%H%M%S')
        device_str = f"{self.target_device.type}{self.target_device.index if self.target_device.index is not None else ''}"
        filename = 'ckpt_{}_{}_{}_{:06d}.tar'.format(self.scene_name, device_str, timestamp, self.global_step)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        if final:
            torch.save({
                'model_d': self.model_d.state_dict(),
                'encoder_d': self.encoder_d.state_dict(),
                'model_v': self.model_v.state_dict(),
                'encoder_v': self.encoder_v.state_dict(),
                'global_step': self.global_step,
                'config': self.config,
                'final': True,
            }, path)
        else:
            torch.save({
                'model_d': self.model_d.state_dict(),
                'encoder_d': self.encoder_d.state_dict(),
                'optimizer_d': self.optimizer_d.state_dict(),
                'model_v': self.model_v.state_dict(),
                'encoder_v': self.encoder_v.state_dict(),
                'optimizer_v': self.optimizer_v.state_dict(),
                'global_step': self.global_step,
                'config': self.config,
                'final': False,
            }, path)
        print(f"Checkpoint saved to {path}")
        return path

    def load_ckpt(self, path: str, device: torch.device):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            assert checkpoint['final'] == False, "Don't load final checkpoint in a Train Model, use it in a Evaluation Model."
            # config = checkpoint['config']
            # config.target_device = device
            # self._reinitialize(config)
            self.model_d.load_state_dict(checkpoint['model_d'])
            self.encoder_d.load_state_dict(checkpoint['encoder_d'])
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            self.model_v.load_state_dict(checkpoint['model_v'])
            self.encoder_v.load_state_dict(checkpoint['encoder_v'])
            self.optimizer_v.load_state_dict(checkpoint['optimizer_v'])
            self.global_step = checkpoint['global_step']
            print(f"loaded checkpoint from {path}")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")


class TrainDensityModel(_TrainModelBase):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

    @torch.compile
    def image_loss(self, batch_indices: torch.Tensor, batch_rays_o: torch.Tensor, batch_rays_d: torch.Tensor, depth_size: int, near: float, far: float):
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

        bbox_mask = insideMask(batch_points_time_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
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

    def forward(self, batch_size: int, depth_size: int):
        self.optimizer_d.zero_grad()
        batch_indices, batch_rays_o, batch_rays_d = self._next_batch(batch_size)
        img_loss = self.image_loss(batch_indices, batch_rays_o, batch_rays_d, depth_size, float(self.near[0].item()), float(self.far[0].item()))
        img_loss.backward()
        self.optimizer_d.step()
        self.scheduler_d.step()
        self.global_step += 1
        return img_loss


class TrainVelocityModel(_TrainModelBase):
    def __init__(self, config: TrainConfig, resx: int, resy: int, resz: int):
        super().__init__(config)
        self.resx, self.resy, self.resz = resx, resy, resz
        self.points_generator = None

    def _next_sampled_points(self, batch_size: int):
        if self.points_generator is None:
            self.points_generator = sample_bbox(self.resx, self.resy, self.resz, batch_size, True, self.target_device, self.target_dtype, self.s2w, self.s_w2s, self.s_scale, self.s_min, self.s_max)
        try:
            return next(self.points_generator)
        except StopIteration:
            self.points_generator = sample_bbox(self.resx, self.resy, self.resz, batch_size, True, self.target_device, self.target_dtype, self.s2w, self.s_w2s, self.s_scale, self.s_min, self.s_max)
            return next(self.points_generator)

    def velocity_loss(self, batch_points):
        def g(x):
            return self.model_d(x)

        batch_time = torch.rand((), device=self.target_device, dtype=self.target_dtype)
        batch_points_time = torch.cat([batch_points, batch_time.expand(batch_points[..., :1].shape)], dim=-1)
        batch_points_time_flat = batch_points_time.reshape(-1, 4)

        bbox_mask = insideMask(batch_points_time_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
        batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]
        batch_points_time_flat_filtered.requires_grad = True

        hidden = self.encoder_d(batch_points_time_flat_filtered)
        raw_d = self.model_d(hidden)

        jac = torch.vmap(torch.func.jacrev(g))(hidden)
        jac_x = get_minibatch_jacobian(hidden, batch_points_time_flat_filtered)
        jac = jac @ jac_x
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]

        raw_vel, raw_f = self.model_v(self.encoder_v(batch_points_time_flat_filtered))
        _u_x, _u_y, _u_z, _u_t = None, None, None, None
        _u, _v, _w = raw_vel.split(1, dim=-1)
        split_nse = _d_t + (_u * _d_x + _v * _d_y + _w * _d_z)
        nse_errors = torch.mean(torch.square(split_nse))
        split_nse_wei = 0.001
        nseloss_fine = nse_errors * split_nse_wei

        proj_loss = torch.zeros_like(nseloss_fine)

        viz_dens_mask = raw_d.detach() > 0.1
        vel_norm = raw_vel.norm(dim=-1, keepdim=True)
        min_vel_mask = vel_norm.detach() < 0.2 * raw_d.detach()
        vel_reg_mask = min_vel_mask & viz_dens_mask
        min_vel_reg_map = (0.2 * raw_d - vel_norm) * vel_reg_mask.float()
        min_vel_reg = min_vel_reg_map.pow(2).mean()

        skip = False
        if nse_errors.sum() > 10000:
            print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
            skip = True
        return skip, nseloss_fine, proj_loss, min_vel_reg

    def forward(self, batch_size: int):
        self.optimizer_v.zero_grad()
        sampled_points = self._next_sampled_points(batch_size)
        skip, nseloss_fine, proj_loss, min_vel_reg = self.velocity_loss(batch_points=sampled_points)
        vel_loss = 1.0 * nseloss_fine + 1.0 * proj_loss + 10.0 * min_vel_reg
        # if not skip: # don't skip
        vel_loss.backward()
        self.optimizer_v.step()
        self.scheduler_v.step()
        self.global_step += 1
        return vel_loss, nseloss_fine, proj_loss, min_vel_reg


class TrainJointModel(_TrainModelBase):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

    def joint_loss(self, batch_indices, batch_rays_o, batch_rays_d, depth_size: int, near: float, far: float):
        def g(x):
            return self.model_d(x)

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

        bbox_mask = insideMask(batch_points_time_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
        batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]
        batch_points_time_flat_filtered.requires_grad = True

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

        jac = torch.vmap(torch.func.jacrev(g))(hidden)
        jac_x = get_minibatch_jacobian(hidden, batch_points_time_flat_filtered)
        jac = jac @ jac_x
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]

        raw_vel, raw_f = self.model_v(self.encoder_v(batch_points_time_flat_filtered))
        _u_x, _u_y, _u_z, _u_t = None, None, None, None
        _u, _v, _w = raw_vel.split(1, dim=-1)
        split_nse = _d_t + (_u * _d_x + _v * _d_y + _w * _d_z)
        nse_errors = torch.mean(torch.square(split_nse))
        split_nse_wei = 0.001
        nseloss_fine = nse_errors * split_nse_wei

        proj_loss = torch.zeros_like(nseloss_fine)

        viz_dens_mask = raw_d.detach() > 0.1
        vel_norm = raw_vel.norm(dim=-1, keepdim=True)
        min_vel_mask = vel_norm.detach() < 0.2 * raw_d.detach()
        vel_reg_mask = min_vel_mask & viz_dens_mask
        min_vel_reg_map = (0.2 * raw_d - vel_norm) * vel_reg_mask.float()
        min_vel_reg = min_vel_reg_map.pow(2).mean()

        skip = False
        if nse_errors.sum() > 10000:
            print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
            skip = True
        return skip, nseloss_fine, img_loss, proj_loss, min_vel_reg

    def forward(self, batch_size: int, depth_size: int):
        self.optimizer_d.zero_grad()
        self.optimizer_v.zero_grad()
        batch_indices, batch_rays_o, batch_rays_d = self._next_batch(batch_size)
        skip, nseloss_fine, img_loss, proj_loss, min_vel_reg = self.joint_loss(batch_indices, batch_rays_o, batch_rays_d, depth_size, float(self.near[0].item()), float(self.far[0].item()))
        vel_loss = 10000 * img_loss + 1.0 * nseloss_fine + 1.0 * proj_loss + 10.0 * min_vel_reg
        # if not skip: # don't skip
        vel_loss.backward()
        self.optimizer_d.step()
        self.optimizer_v.step()
        self.scheduler_d.step()
        self.scheduler_v.step()
        self.global_step += 1
        return vel_loss, nseloss_fine, img_loss, proj_loss, min_vel_reg
