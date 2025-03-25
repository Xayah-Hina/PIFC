from lib.dataset import *
from lib.frustum import *
from model.encoder_hyfluid import *
from model.model_hyfluid import *

import math
import tqdm


class HoudiniExecutor:
    def __init__(self, batch_size: int, depth_size: int, ratio: float, target_device: torch.device, target_dtype: torch.dtype):
        videos_data = load_videos_data(*training_videos, ratio=ratio, dtype=target_dtype)
        poses, focals, width, height, near, far = load_cameras_data(*camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)
        encoder_d = HashEncoderNativeFasterBackward().to(target_device)
        model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_d.num_levels * 2).to(target_device)
        optimizer_d = torch.optim.RAdam([{'params': model_d.parameters(), 'weight_decay': 1e-6}, {'params': encoder_d.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))
        encoder_v = HashEncoderNativeFasterBackward().to(target_device)
        model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=encoder_v.num_levels * 2, use_f=False).to(target_device)
        optimizer_v = torch.optim.RAdam([{'params': model_v.parameters(), 'weight_decay': 1e-6}, {'params': encoder_v.parameters(), 'eps': 1e-15}], lr=0.001, betas=(0.9, 0.99))

        self.target_device = target_device
        self.target_dtype = target_dtype
        self.encoder_d = encoder_d
        self.model_d = model_d
        self.optimizer_d = optimizer_d
        self.encoder_v = encoder_v
        self.model_v = model_v
        self.optimizer_v = optimizer_v
        target_lr_ratio = 0.1
        gamma = math.exp(math.log(target_lr_ratio) / 30000)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d, gamma=gamma)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, gamma=gamma)

        self.videos_data = videos_data
        self.poses, self.focals, self.width, self.height, self.near, self.far = poses, focals, width, height, near, far
        self.generator = None
        self.videos_data_resampled = None

        self.batch_size = batch_size
        self.depth_size = depth_size
        self.global_step = 1
        self.NEAR_float = float(near[0].item())
        self.FAR_float = float(far[0].item())

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

        self.refresh_generator()

    def refresh_generator(self):
        dirs, u, v = shuffle_uv(focals=self.focals, width=int(self.width[0].item()), height=int(self.height[0].item()), randomize=True, device=torch.device("cpu"), dtype=self.target_dtype)
        dirs = dirs.to(self.target_device)
        self.videos_data_resampled = resample_frames(frames=self.videos_data, u=u, v=v).to(self.target_device)  # (T, V, H, W, C)
        self.generator = sample_frustum(dirs=dirs, poses=self.poses, batch_size=self.batch_size, randomize=True, device=self.target_device)

    def get_mask(self, batch_points):
        return insideMask(batch_points, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)

    def optimize_density(self):
        self.optimizer_d.zero_grad()
        try:
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        except StopIteration:
            self.refresh_generator()
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        img_loss = self.density_only_loss(batch_indices, batch_rays_o, batch_rays_d)
        img_loss.backward()
        self.optimizer_d.step()
        self.scheduler_d.step()
        self.global_step += 1
        tqdm.tqdm.write(f"iter: {self.global_step}, lr_d: {self.scheduler_d.get_last_lr()[0]}, img_loss: {img_loss}")

    def optimize_joint(self):
        self.optimizer_d.zero_grad()
        self.optimizer_v.zero_grad()
        try:
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        except StopIteration:
            self.refresh_generator()
            batch_indices, batch_rays_o, batch_rays_d = next(self.generator)
        skip, nseloss_fine, img_loss, proj_loss, min_vel_reg = self.velocity_loss(batch_indices, batch_rays_o, batch_rays_d)
        if not skip:
            vel_loss = nseloss_fine + 10000 * img_loss + 1.0 * proj_loss + 10.0 * min_vel_reg
            vel_loss.backward()
            self.optimizer_d.step()
            self.optimizer_v.step()
            self.scheduler_d.step()
            self.scheduler_v.step()
            self.global_step += 1
            tqdm.tqdm.write(f"iter: {self.global_step}, lr_d: {self.scheduler_d.get_last_lr()[0]}, lr_v: {self.scheduler_v.get_last_lr()[0]}, nseloss_fine: {nseloss_fine}, img_loss: {img_loss}, proj_loss: {proj_loss}, min_vel_reg: {min_vel_reg}")

    @torch.compile
    def density_only_loss(self, batch_indices, batch_rays_o, batch_rays_d):
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

        bbox_mask = self.get_mask(batch_points_time_flat[..., :3])
        batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]

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

        return img_loss

    def velocity_loss(self, batch_indices, batch_rays_o, batch_rays_d):
        def g(x):
            return self.model_d(x)

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

        bbox_mask = self.get_mask(batch_points_time_flat[..., :3])
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

    @torch.compile
    def sample_density_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            bbox_mask = self.get_mask(input_xyzt_flat[..., :3])

            raw_d_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d_flat[~bbox_mask] = 0.0
            raw_d = raw_d_flat.reshape(resx, resy, resz, 1)
            return raw_d.to(torch.device('cpu'))

    @torch.compile
    def sample_velocity_grid(self, resx, resy, resz, frame):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            input_xyzt_flat = torch.cat([coord_3d_world, torch.ones_like(coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            bbox_mask = self.get_mask(input_xyzt_flat[..., :3])

            raw_vel_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_vel_flat_batch, _ = self.model_v(self.encoder_v(input_xyzt_flat_batch))
                raw_vel_flat_list.append(raw_vel_flat_batch)
            raw_vel_flat = torch.cat(raw_vel_flat_list, dim=0)
            raw_vel_flat[~bbox_mask] = 0.0
            raw_vel = raw_vel_flat.reshape(resx, resy, resz, 3)
            return raw_vel.to(torch.device('cpu'))

    @torch.compile
    def sample_points(self, resx, resy, resz):
        with torch.no_grad():
            xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device), torch.linspace(0, 1, resy, device=self.target_device), torch.linspace(0, 1, resz, device=self.target_device)], indexing='ij')
            coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
            coord_3d_world = sim2world(coord_3d_sim, self.s2w, self.s_scale)
            return coord_3d_world.to(torch.device('cpu'))


def run_sample_grids(ckpt_path):
    torch.set_float32_matmul_precision('high')
    executor = HoudiniExecutor(batch_size=1024, depth_size=192, ratio=0.5, target_device=torch.device("cuda:0"), target_dtype=torch.float32)
    executor.load_ckpt(ckpt_path)
    raw_d = executor.sample_density_grid(64, 64, 64, 0)
    raw_vel = executor.sample_velocity_grid(64, 64, 64, 0)
    print(raw_d.shape, raw_vel.shape)


def run_train_density_only():
    torch.set_float32_matmul_precision('high')
    executor = HoudiniExecutor(batch_size=1024, depth_size=192, ratio=0.5, target_device=torch.device("cuda:0"), target_dtype=torch.float32)
    pbar = tqdm.tqdm(desc="run_train_density_only", unit="iter")
    try:
        for _ in range(1000):
            executor.optimize_density()
            pbar.update(1)
    except Exception as e:
        print(e)
    finally:
        executor.save_ckpt('houdini/ckpt_den_only')


def run_train_joint():
    torch.set_float32_matmul_precision('high')
    executor = HoudiniExecutor(batch_size=1024, depth_size=192, ratio=0.5, target_device=torch.device("cuda:0"), target_dtype=torch.float32)
    pbar = tqdm.tqdm(desc="run_train_joint", unit="iter")
    try:
        for _ in range(1000):
            executor.optimize_joint()
            pbar.update(1)
    except Exception as e:
        print(e)
    finally:
        executor.save_ckpt('houdini/ckpt_joint')


if __name__ == '__main__':
    # run_train_density_only()
    run_train_joint()
    # run_sample_grids('houdini/ckpt/den_002533.tar')
