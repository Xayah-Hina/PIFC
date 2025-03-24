from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *


class HoudiniExecutor:
    def __init__(self, batch_size: int, depth_size: int, ratio: float, target_device: torch.device, target_dtype: torch.dtype, start_ckpt=None):
        videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
        poses, focals, width, height, near, far = load_cameras_data(*camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)
        encoder_d = HashEncoderNativeFasterBackward().to(target_device)
        model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_d.num_levels * 2).to(target_device)
        optimizer_d = torch.optim.RAdam([{'params': model_d.parameters(), 'weight_decay': 1e-6}, {'params': encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
        encoder_v = HashEncoderNativeFasterBackward().to(target_device)
        model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_v.num_levels * 2, use_f=False).to(target_device)
        optimizer_v = torch.optim.RAdam([{'params': model_v.parameters(), 'weight_decay': 1e-6}, {'params': encoder_v.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        # =============================================================================
        # TO DELETE
        VOXEL_TRAN = torch.tensor([
            [1.0, 0.0, 7.5497901e-08, 8.1816666e-02],
            [0.0, 1.0, 0.0, -4.4627272e-02],
            [7.5497901e-08, 0.0, -1.0, -4.9089999e-03],
            [0.0, 0.0, 0.0, 1.0]
        ], device=target_device, dtype=target_dtype)

        VOXEL_SCALE = torch.tensor([0.4909, 0.73635, 0.4909], device=target_device, dtype=target_dtype)

        NEAR_float, FAR_float = float(near[0].item()), float(far[0].item())

        self.s_w2s = torch.inverse(VOXEL_TRAN).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = VOXEL_SCALE.expand([3])
        self.s_min = torch.tensor([0.15, 0.0, 0.15], device=target_device, dtype=target_dtype)
        self.s_max = torch.tensor([0.85, 1.0, 0.85], device=target_device, dtype=target_dtype)

        # TO DELETE
        # =============================================================================

        dirs, u, v = shuffle_uv(focals=focals, width=int(width[0].item()), height=int(height[0].item()), randomize=True, device=torch.device("cpu"), dtype=target_dtype)
        videos_data_resampled = resample_frames(frames=videos_data, u=u, v=v).to(target_device)  # (T, V, H, W, C)
        dirs = dirs.to(target_device)

        self.generator = sample_frustum(dirs=dirs, poses=poses, batch_size=batch_size, randomize=True, device=target_device)
        self.encoder_d = encoder_d
        self.model_d = model_d
        self.optimizer_d = optimizer_d
        self.encoder_v = encoder_v
        self.model_v = model_v
        self.optimizer_v = optimizer_v
        self.videos_data_resampled = videos_data_resampled
        self.target_device = target_device
        self.target_dtype = target_dtype

        self.NEAR_float = NEAR_float
        self.FAR_float = FAR_float
        self.batch_size = batch_size
        self.depth_size = depth_size

        self.global_step = 1
        self.lrate = 0.01
        self.lrate_decay = 1000

    def forward_1_iter(self):
        def g(x):
            return self.model_d(x)

        self.optimizer_d.zero_grad()
        self.optimizer_v.zero_grad()

        try:
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        except StopIteration:
            return False
        batch_time, batch_target_pixels = sample_random_frame(videos_data=self.videos_data_resampled, batch_indices=batch_indices, device=self.target_device, dtype=self.target_dtype)

        batch_size_current = batch_rays_d.shape[0]

        t_vals = torch.linspace(0., 1., steps=self.depth_size, device=self.target_device, dtype=self.target_dtype)
        t_vals = t_vals.view(1, self.depth_size)
        z_vals = self.NEAR_float * (1. - t_vals) + self.FAR_float * t_vals
        z_vals = z_vals.expand(batch_size_current, self.depth_size)

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
        raw_flat = torch.zeros([batch_size_current * self.depth_size], device=self.target_device, dtype=self.target_dtype)
        raw_d_flat = raw_d.view(-1)
        raw_flat = raw_flat.masked_scatter(bbox_mask, raw_d_flat)
        raw = raw_flat.reshape(batch_size_current, self.depth_size, 1)

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

        if nse_errors.sum() > 10000:
            print(f'skip large loss {nse_errors.sum():.3g}, timestep={batch_points_time_flat_filtered[0, 3]}')
            return True

        vel_loss = nseloss_fine + 10000 * img_loss + 1.0 * proj_loss + 10.0 * min_vel_reg
        vel_loss.backward()

        self.optimizer_d.step()
        self.optimizer_v.step()

        decay_rate = 0.1
        decay_steps = self.lrate_decay
        new_lrate = self.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = new_lrate
        for param_group in self.optimizer_v.param_groups:
            param_group['lr'] = new_lrate

        self.global_step += 1

        print(f"iter: {self.global_step}, nseloss_fine: {nseloss_fine}, img_loss: {10000 * img_loss}, proj_loss: {proj_loss}, min_vel_reg: {10.0 * min_vel_reg}, vel_loss: {vel_loss}")

        return True

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
        print(f"loaded checkpoint from {path}")

    def sample_density_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            raw_d_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d = raw_d_flat.reshape(resx, resy, resz, 1)
            return raw_d.to(torch.device('cpu'))

    def sample_velocity_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            raw_vel_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_vel_flat_batch, _ = self.model_v(self.encoder_v(input_xyzt_flat_batch))
                raw_vel_flat_list.append(raw_vel_flat_batch)
            raw_vel_flat = torch.cat(raw_vel_flat_list, dim=0)
            raw_vel = raw_vel_flat.reshape(resx, resy, resz, 3)
            return raw_vel.to(torch.device('cpu'))

    def sample_points(self, resx, resy, resz):
        xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
        coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
        coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
        return coord_3d_world.to(torch.device('cpu'))


import tqdm

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    executor = HoudiniExecutor(batch_size=1024, depth_size=192, ratio=0.5, target_device=torch.device("cuda:0"), target_dtype=torch.float32)
    pbar = tqdm.tqdm(desc="Running forward_1_iter", unit="iter")
    while executor.forward_1_iter():
        pbar.update(1)
    pbar.close()
    executor.save_ckpt('ckpt')
