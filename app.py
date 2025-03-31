from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *

import torch
import math
import tqdm
import os


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
        self.encoder_d = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        self.optimizer_d = torch.optim.RAdam([{'params': self.model_d.parameters(), 'weight_decay': 1e-6}, {'params': self.encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
        self.encoder_v = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
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
        return img_loss

    def optimize_velocity(self, batch_points):
        self.optimizer_v.zero_grad()
        skip, nseloss_fine, proj_loss, min_vel_reg = self.velocity_loss(batch_points)
        if not skip:
            vel_loss = 1.0 * nseloss_fine + 1.0 * proj_loss + 10.0 * min_vel_reg
            vel_loss.backward()
            self.optimizer_v.step()
            self.scheduler_v.step()
            self.global_step += 1
            tqdm.tqdm.write(f"iter: {self.global_step}, lr_d: {self.scheduler_d.get_last_lr()[0]}, lr_v: {self.scheduler_v.get_last_lr()[0]}, nseloss_fine: {nseloss_fine}, proj_loss: {proj_loss}, min_vel_reg: {min_vel_reg}")
        return nseloss_fine, proj_loss, min_vel_reg

    def optimize_joint(self, depth_size: int):
        self.optimizer_d.zero_grad()
        self.optimizer_v.zero_grad()
        try:
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        except StopIteration:
            self.generator, self.videos_data_resampled = refresh_generator(self.batch_size, self.videos_data, self.poses, self.focals, int(self.width[0].item()), int(self.height[0].item()), self.target_device, self.target_dtype)
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        skip, nseloss_fine, img_loss, proj_loss, min_vel_reg = self.joint_loss(batch_indices, batch_rays_o, batch_rays_d, depth_size, float(self.near[0].item()), float(self.far[0].item()))
        if not skip:
            vel_loss = nseloss_fine + 10000 * img_loss + 1.0 * proj_loss + 10.0 * min_vel_reg
            vel_loss.backward()
            self.optimizer_d.step()
            self.optimizer_v.step()
            self.scheduler_d.step()
            self.scheduler_v.step()
            self.global_step += 1
            tqdm.tqdm.write(f"iter: {self.global_step}, lr_d: {self.scheduler_d.get_last_lr()[0]}, lr_v: {self.scheduler_v.get_last_lr()[0]}, nseloss_fine: {nseloss_fine}, img_loss: {img_loss}, proj_loss: {proj_loss}, min_vel_reg: {min_vel_reg}")
        return nseloss_fine, img_loss, proj_loss, min_vel_reg

    @torch.compile
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
        # min_vel_reg = torch.zeros_like(nseloss_fine)

        skip = False
        if nse_errors.sum() > 10000:
            print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
            skip = True
        return skip, nseloss_fine, proj_loss, min_vel_reg

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
        # min_vel_reg = torch.zeros_like(nseloss_fine)

        skip = False
        if nse_errors.sum() > 10000:
            print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
            skip = True
        return skip, nseloss_fine, img_loss, proj_loss, min_vel_reg

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


class ValidationModel:
    def __init__(self, target_device: torch.device, target_dtype: torch.dtype):
        self._load_model(target_device)
        self._load_valid_domain(target_device, target_dtype)
        self.target_device = target_device
        self.target_dtype = target_dtype

    def _load_model(self, target_device: torch.device):
        self.encoder_d = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        self.encoder_v = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_v.num_levels * 2, use_f=False).to(target_device)

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

    def load_ckpt(self, path: str):
        checkpoint = torch.load(path)
        self.model_d.load_state_dict(checkpoint['model_d'])
        self.encoder_d.load_state_dict(checkpoint['encoder_d'])
        self.model_v.load_state_dict(checkpoint['model_v'])
        self.encoder_v.load_state_dict(checkpoint['encoder_v'])
        tqdm.tqdm.write(f"loaded checkpoint from {path}")

    @torch.compile
    def sample_density_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resy, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resz, device=self.target_device, dtype=self.target_dtype)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            bbox_mask = insideMask(input_xyzt_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)

            raw_d_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d_flat[~bbox_mask] = 0.0
            raw_d = raw_d_flat.reshape(resx, resy, resz, 1)
            return raw_d

    @torch.compile
    def sample_velocity_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resy, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resz, device=self.target_device, dtype=self.target_dtype)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            bbox_mask = insideMask(input_xyzt_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)

            raw_vel_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_vel_flat_batch, _ = self.model_v(self.encoder_v(input_xyzt_flat_batch))
                raw_vel_flat_list.append(raw_vel_flat_batch)
            raw_vel_flat = torch.cat(raw_vel_flat_list, dim=0)
            raw_vel_flat[~bbox_mask] = 0.0
            raw_vel = raw_vel_flat.reshape(resx, resy, resz, 3)
            return raw_vel

    def resimulation(self, resx, resy, resz, dt):
        with torch.no_grad():
            source_height = 0.15


def train_density_only(total_iter, batch_size, depth_size, ratio, target_device, target_dtype, pretrained_ckpt=None):
    losses = []
    steps = []
    avg_losses = []
    avg_steps = []

    model = TrainModel(batch_size, ratio, target_device, target_dtype)
    if pretrained_ckpt:
        model.load_ckpt(pretrained_ckpt)
    try:
        for _ in tqdm.trange(total_iter):
            img_loss = model.optimize_density(depth_size)
            losses.append(img_loss.item())
            steps.append(model.global_step)

            if len(steps) % 100 == 0:
                avg_loss = sum(losses[-100:]) / 100
                avg_losses.append(avg_loss)
                avg_steps.append(model.global_step)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt('ckpt/train_density_only')

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%H")
        filename = f"loss_train_density_only_{timestamp}_bs{batch_size}_{model.global_step}.png"
        save_dir = 'ckpt/image'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(avg_steps, avg_losses, label='Average Image Loss (every 100 steps)', color='blue', linestyle='-', marker='o')
        plt.xlabel('Global Step')
        plt.ylabel('Image Loss (Averaged)')
        plt.title('Average Image Loss vs. Global Step')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.savefig(save_path)
        plt.close()

        print(f"Image loss curve saved to {save_path}")


def train_velocity_only(total_iter, batch_size, ratio, target_device, target_dtype, pretrained_ckpt=None):
    losses_nseloss_fine = []
    losses_proj = []
    losses_min_vel_reg = []
    losses = []
    steps = []
    avg_losses_nseloss_fine = []
    avg_losses_proj = []
    avg_losses_min_vel_reg = []
    ave_losses = []
    avg_steps = []

    model = TrainModel(batch_size, ratio, target_device, target_dtype)
    if pretrained_ckpt:
        model.load_ckpt(pretrained_ckpt)
    try:
        points_generator = sample_bbox(128, 192, 128, 1024, True, target_device, target_dtype, model.s2w, model.s_w2s, model.s_scale, model.s_min, model.s_max)
        for _ in tqdm.trange(total_iter):
            try:
                batch_points = next(points_generator)
            except StopIteration:
                points_generator = sample_bbox(128, 192, 128, 1024, True, target_device, target_dtype, model.s2w, model.s_w2s, model.s_scale, model.s_min, model.s_max)
                batch_points = next(points_generator)
            nseloss_fine, proj_loss, min_vel_reg = model.optimize_velocity(batch_points)
            losses_nseloss_fine.append(nseloss_fine.item())
            losses_proj.append(proj_loss.item())
            losses_min_vel_reg.append(min_vel_reg.item())
            losses.append(nseloss_fine.item() + proj_loss.item() + min_vel_reg.item())
            steps.append(model.global_step)

            if len(steps) % 100 == 0:
                avg_loss_nseloss_fine = sum(losses_nseloss_fine[-100:]) / 100
                avg_loss_proj = sum(losses_proj[-100:]) / 100
                avg_loss_min_vel_reg = sum(losses_min_vel_reg[-100:]) / 100
                avg_loss = sum(losses[-100:]) / 100
                avg_losses_nseloss_fine.append(avg_loss_nseloss_fine)
                avg_losses_proj.append(avg_loss_proj)
                avg_losses_min_vel_reg.append(avg_loss_min_vel_reg)
                ave_losses.append(avg_loss)
                avg_steps.append(model.global_step)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt('ckpt/train_velocity_only')
        from datetime import datetime
        import os
        import matplotlib.pyplot as plt

        # 获取时间戳和保存目录
        timestamp = datetime.now().strftime("%m%d%H")
        save_dir = 'ckpt/image'
        os.makedirs(save_dir, exist_ok=True)

        # 定义四个损失及其属性
        loss_data = [
            ("nseloss_fine", avg_steps, avg_losses_nseloss_fine, 'red', 'Average NSE Loss Fine (every 100 steps)'),
            ("proj_loss", avg_steps, avg_losses_proj, 'green', 'Average Proj Loss (every 100 steps)'),
            ("min_vel_reg", avg_steps, avg_losses_min_vel_reg, 'purple', 'Average Min Vel Reg Loss (every 100 steps)'),
            ('total_loss', avg_steps, ave_losses, 'blue', 'Average Total Loss (every 100 steps)')
        ]

        # 绘制并保存四个独立的损失曲线
        for loss_name, steps, losses, color, title in loss_data:
            # 构建文件名
            filename = f"loss_{loss_name}_{timestamp}_bs{batch_size}_{model.global_step}.png"
            save_path = os.path.join(save_dir, filename)

            # 绘制曲线
            plt.figure(figsize=(8, 6))
            plt.plot(steps, losses, label=title, color=color, linestyle='-', marker='o')
            plt.xlabel('Global Step')
            plt.ylabel('Loss (Averaged)')
            plt.title(f'{title} vs. Global Step')
            plt.legend()
            plt.grid(True)
            plt.show()

            # 保存图片
            plt.savefig(save_path)
            plt.close()

            print(f"{loss_name} loss curve saved to {save_path}")


def train_joint(total_iter, batch_size, depth_size, ratio, target_device, target_dtype, pretrained_ckpt=None):
    losses_nseloss_fine = []
    losses_img = []
    losses_proj = []
    losses_min_vel_reg = []
    steps = []
    avg_losses_nseloss_fine = []
    avg_losses_img = []
    avg_losses_proj = []
    avg_losses_min_vel_reg = []
    avg_steps = []

    model = TrainModel(batch_size, ratio, target_device, target_dtype)
    if pretrained_ckpt:
        model.load_ckpt(pretrained_ckpt)
    try:
        for _ in tqdm.trange(total_iter):
            nseloss_fine, img_loss, proj_loss, min_vel_reg = model.optimize_joint(depth_size)
            losses_nseloss_fine.append(nseloss_fine.item())
            losses_img.append(img_loss.item())
            losses_proj.append(proj_loss.item())
            losses_min_vel_reg.append(min_vel_reg.item())
            steps.append(model.global_step)

            if len(steps) % 100 == 0:
                avg_loss_nseloss_fine = sum(losses_nseloss_fine[-100:]) / 100
                avg_loss_img = sum(losses_img[-100:]) / 100
                avg_loss_proj = sum(losses_proj[-100:]) / 100
                avg_loss_min_vel_reg = sum(losses_min_vel_reg[-100:]) / 100
                avg_losses_nseloss_fine.append(avg_loss_nseloss_fine)
                avg_losses_img.append(avg_loss_img)
                avg_losses_proj.append(avg_loss_proj)
                avg_losses_min_vel_reg.append(avg_loss_min_vel_reg)
                avg_steps.append(model.global_step)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt('ckpt/train_joint')
        from datetime import datetime
        import os
        import matplotlib.pyplot as plt

        # 获取时间戳和保存目录
        timestamp = datetime.now().strftime("%m%d%H")
        save_dir = 'ckpt/image'
        os.makedirs(save_dir, exist_ok=True)

        # 定义四个损失及其属性
        loss_data = [
            ("nseloss_fine", avg_steps, avg_losses_nseloss_fine, 'red', 'Average NSE Loss Fine (every 100 steps)'),
            ("img_loss", avg_steps, avg_losses_img, 'blue', 'Average Image Loss (every 100 steps)'),
            ("proj_loss", avg_steps, avg_losses_proj, 'green', 'Average Proj Loss (every 100 steps)'),
            ("min_vel_reg", avg_steps, avg_losses_min_vel_reg, 'purple', 'Average Min Vel Reg Loss (every 100 steps)')
        ]

        # 绘制并保存四个独立的损失曲线
        for loss_name, steps, losses, color, title in loss_data:
            # 构建文件名
            filename = f"loss_{loss_name}_{timestamp}_bs{batch_size}_{model.global_step}.png"
            save_path = os.path.join(save_dir, filename)

            # 绘制曲线
            plt.figure(figsize=(8, 6))
            plt.plot(steps, losses, label=title, color=color, linestyle='-', marker='o')
            plt.xlabel('Global Step')
            plt.ylabel('Loss (Averaged)')
            plt.title(f'{title} vs. Global Step')
            plt.legend()
            plt.grid(True)
            plt.show()

            # 保存图片
            plt.savefig(save_path)
            plt.close()

            print(f"{loss_name} loss curve saved to {save_path}")


def validate_sample_grid(resx, resy, resz, target_device, target_dtype, ckpt_path):
    model = ValidationModel(target_device, target_dtype)
    model.load_ckpt(ckpt_path)
    os.makedirs('ckpt/sampled_grid', exist_ok=True)
    for frame in tqdm.trange(120):
        raw_d = model.sample_density_grid(resx, resy, resz, frame)
        raw_v = model.sample_velocity_grid(resx, resy, resz, frame)
        np.savez_compressed(f'ckpt/sampled_grid/sampled_grid_{frame + 1:03d}.npz', den=raw_d.cpu().numpy(), vel=raw_v.cpu().numpy())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['train_density_only', 'train_velocity_only', 'train_joint', 'validate_sample_grid'], required=True, help="Choose the operation to execute.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the operation.")
    args = parser.parse_args()

    if args.option == "train_density_only":
        train_density_only(
            total_iter=1000,
            batch_size=1024,
            depth_size=192,
            ratio=0.5,
            target_device=torch.device(args.device),
            target_dtype=torch.float32,
            pretrained_ckpt=args.ckpt_path,
        )

    if args.option == "train_velocity_only":
        train_velocity_only(
            total_iter=1000,
            batch_size=1024,
            ratio=0.5,
            target_device=torch.device(args.device),
            target_dtype=torch.float32,
            pretrained_ckpt=args.ckpt_path,
        )

    if args.option == "train_joint":
        train_joint(
            total_iter=1000,
            batch_size=1024,
            depth_size=192,
            ratio=0.5,
            target_device=torch.device(args.device),
            target_dtype=torch.float32,
            pretrained_ckpt=args.ckpt_path,
        )

    if args.option == "validate_sample_grid":
        validate_sample_grid(
            resx=128,
            resy=192,
            resz=128,
            target_device=torch.device(args.device),
            target_dtype=torch.float32,
            ckpt_path=args.ckpt_path,
        )
