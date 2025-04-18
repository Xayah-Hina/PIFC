from .utils.dataset import *
from .utils.frustum import *
from .utils.solver import *
from .model.encoder_hyfluid import *
from .model.model_hyfluid import *
import torch
import dataclasses


@dataclasses.dataclass
class EvaluationConfig:
    # checkpoint p  ath
    pretrained_ckpt: str
    scene_name: str

    # general parameters
    target_device: torch.device
    target_dtype: torch.dtype

    ratio: float

    use_rgb: bool

    frame_start: int
    frame_end: int

    def __post_init__(self):
        import yaml
        import os
        scene_info_path = f'data/{self.scene_name}/scene_info.yaml'
        assert os.path.exists(scene_info_path), f"Scene info file not found: {scene_info_path}"
        with open(scene_info_path, 'r') as f:
            scene_info = yaml.safe_load(f)
            self.validation_videos = scene_info['validation_videos']
            self.validation_camera_calibrations = scene_info['validation_camera_calibrations']
            self.voxel_transform = torch.tensor(scene_info['voxel_transform'], device=self.target_device, dtype=self.target_dtype)
            self.voxel_scale = torch.tensor(scene_info['voxel_scale'], device=self.target_device, dtype=self.target_dtype)
            self.s_min = torch.tensor(scene_info['s_min'], device=self.target_device, dtype=self.target_dtype)
            self.s_max = torch.tensor(scene_info['s_max'], device=self.target_device, dtype=self.target_dtype)
        self.s_w2s = torch.inverse(self.voxel_transform).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = self.voxel_scale.expand([3])


class _EvaluationModelBase:
    def __init__(self, config):
        self._load_model(config.target_device, config.use_rgb)
        self._load_validation_dataset(config.validation_videos, config.validation_camera_calibrations, config.frame_start, config.frame_end, config.ratio, config.target_device, config.target_dtype)
        self.load_ckpt(config.pretrained_ckpt, config.target_device)

        self.target_device = config.target_device
        self.target_dtype = config.target_dtype
        self.s_w2s, self.s2w, self.s_scale, self.s_min, self.s_max = config.s_w2s, config.s2w, config.s_scale, config.s_min, config.s_max
        self.ratio = config.ratio

    def _load_model(self, target_device: torch.device, use_rgb):
        self.encoder_d = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        if use_rgb:
            self.model_d = NeRFSmall_c(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        else:
            self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2).to(target_device)
        self.encoder_v = HashEncoderNativeFasterBackward(device=target_device).to(target_device)
        self.model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_v.num_levels * 2, use_f=False).to(target_device)

    def load_ckpt(self, path: str, device: torch.device):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.model_d.load_state_dict(checkpoint['model_d'])
            self.encoder_d.load_state_dict(checkpoint['encoder_d'])
            self.model_v.load_state_dict(checkpoint['model_v'])
            self.encoder_v.load_state_dict(checkpoint['encoder_v'])
            print(f"loaded checkpoint from {path}")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _load_validation_dataset(self, validation_videos, validation_camera_calibrations, frame_start: int, frame_end: int, ratio: float, target_device: torch.device, target_dtype: torch.dtype):
        assert len(validation_videos) == len(validation_camera_calibrations), "Number of videos and camera calibrations must match."
        self.videos_data_validation = load_videos_data(*validation_videos, ratio=ratio, dtype=target_dtype)[frame_start:frame_end].to(target_device)
        self.poses_validation, self.focals_validation, self.width_validation, self.height_validation, self.near_validation, self.far_validation = load_cameras_data(*validation_camera_calibrations, ratio=ratio, device=target_device, dtype=target_dtype)

        self.width = int(self.width_validation[0].item())
        self.height = int(self.height_validation[0].item())
        self.near = float(self.near_validation[0].item())
        self.far = float(self.far_validation[0].item())


class EvaluationRenderFrame(_EvaluationModelBase):
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)

    @torch.compile
    def render_frame(self, batch_ray_size: int, depth_size: int, frame: int):
        pose = self.poses_validation[0]
        focal = self.focals_validation[0]
        with torch.no_grad():
            poses = pose.unsqueeze(0)
            focals = focal.unsqueeze(0)
            frame_time = torch.tensor(frame / 120.0, device=self.target_device, dtype=self.target_dtype)

            dirs, _, _ = shuffle_uv(focals=focals, width=self.width, height=self.height, randomize=False, device=self.target_device, dtype=self.target_dtype)
            rays_d = torch.einsum('nij,nhwj->nhwi', poses[:, :3, :3], dirs)  # (1, H, W, 3)
            rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # (1, H, W, 3)
            rays_d = rays_d.reshape(-1, 3)  # (1*H*W, 3)
            rays_o = rays_o.reshape(-1, 3)  # (1*H*W, 3)
            total_ray_size = rays_d.shape[0]

            final_rgb_map_list = []
            for start_ray_index in range(0, total_ray_size, batch_ray_size):
                batch_rays_d = rays_d[start_ray_index:start_ray_index + batch_ray_size]
                batch_rays_o = rays_o[start_ray_index:start_ray_index + batch_ray_size]
                batch_size_current = batch_rays_d.shape[0]

                t_vals = torch.linspace(0., 1., steps=depth_size, device=self.target_device, dtype=self.target_dtype)
                t_vals = t_vals.view(1, depth_size)
                z_vals = self.near * (1. - t_vals) + self.far * t_vals
                z_vals = z_vals.expand(batch_size_current, depth_size)

                batch_dist_vals = z_vals[..., 1:] - z_vals[..., :-1]  # [batch_size_current, N_depths-1]
                batch_points = batch_rays_o[:, None, :] + batch_rays_d[:, None, :] * z_vals[..., :, None]  # [batch_size_current, N_depths, 3]
                batch_points_time = torch.cat([batch_points, frame_time.expand(batch_points[..., :1].shape)], dim=-1)
                batch_points_time_flat = batch_points_time.reshape(-1, 4)

                bbox_mask = insideMask(batch_points_time_flat[..., :3], self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
                batch_points_time_flat_filtered = batch_points_time_flat[bbox_mask]

                hidden = self.encoder_d(batch_points_time_flat_filtered)
                raw_d = self.model_d(hidden)
                raw_flat = torch.zeros([batch_size_current * depth_size], device=self.target_device, dtype=self.target_dtype)
                raw_d_flat = raw_d[..., 0].view(-1)
                raw_flat = raw_flat.masked_scatter(bbox_mask, raw_d_flat)
                raw = raw_flat.reshape(batch_size_current, depth_size, 1)

                dists_cat = torch.cat([batch_dist_vals, torch.tensor([1e10], device=self.target_device).expand(batch_dist_vals[..., :1].shape)], -1)  # [batch_size_current, N_depths]
                dists_final = dists_cat * torch.norm(batch_rays_d[..., None, :], dim=-1)  # [batch_size_current, N_depths]

                noise = 0.
                alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1] + noise) * dists_final)
                weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.target_device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

                if raw_d.shape[-1] == 1:
                    rgb_trained = torch.ones(3, device=self.target_device) * (0.6 + torch.tanh(self.model_d.rgb) * 0.4)
                    rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)
                else:
                    rgb_flat = torch.zeros([batch_size_current * depth_size, 3], device=self.target_device, dtype=self.target_dtype)
                    rgb_valid_flat = raw_d[..., 1:].reshape(-1, 3)
                    for i in range(3):
                        rgb_flat[:, i] = rgb_flat[:, i].masked_scatter(bbox_mask, rgb_valid_flat[:, i])
                    rgb = torch.sigmoid(rgb_flat.reshape(batch_size_current, depth_size, 3))
                    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

                final_rgb_map_list.append(rgb_map)

            final_rgb_map = torch.cat(final_rgb_map_list, dim=0).reshape(self.height, self.width, 3)
            return final_rgb_map


class EvaluationResimulation(_EvaluationModelBase):
    def __init__(self, config: EvaluationConfig, resx: int, resy: int, resz: int):
        super().__init__(config)

        xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resy, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resz, device=self.target_device, dtype=self.target_dtype)], indexing='ij')
        self.coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
        self.coord_3d_world = sim2world(self.coord_3d_sim, self.s2w, self.s_scale)
        self.bbox_mask = insideMask(self.coord_3d_world, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
        self.bbox_mask_flat = self.bbox_mask.flatten()

        self.resx, self.resy, self.resz = resx, resy, resz

    @torch.compile
    def sample_density_grid(self, frame):
        with torch.no_grad():
            input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            raw_d_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))[..., 0]
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d_flat[~self.bbox_mask_flat] = 0.0
            raw_d = raw_d_flat.reshape(self.resx, self.resy, self.resz, 1)
            return raw_d

    @torch.compile
    def sample_velocity_grid(self, frame):
        with torch.no_grad():
            input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            raw_vel_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_vel_flat_batch, _ = self.model_v(self.encoder_v(input_xyzt_flat_batch))
                raw_vel_flat_list.append(raw_vel_flat_batch)
            raw_vel_flat = torch.cat(raw_vel_flat_list, dim=0)
            raw_vel_flat[~self.bbox_mask_flat] = 0.0
            raw_vel = raw_vel_flat.reshape(self.resx, self.resy, self.resz, 3)
            return raw_vel

    @torch.compile
    def advect_density(self, den, vel, source, dt, mask_to_sim):
        with torch.no_grad():
            den = advect_maccormack(den, vel, self.coord_3d_sim, dt)
            den[~mask_to_sim] = source[~mask_to_sim]
            den[~self.bbox_mask] *= 0.
            return den


class EvaluationSpatialState(_EvaluationModelBase):
    def __init__(self, config: EvaluationConfig, resx: int, resy: int, resz: int):
        super().__init__(config)

        xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resy, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resz, device=self.target_device, dtype=self.target_dtype)], indexing='ij')
        self.coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
        self.coord_3d_world = sim2world(self.coord_3d_sim, self.s2w, self.s_scale)
        self.bbox_mask = insideMask(self.coord_3d_world, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
        self.bbox_mask_flat = self.bbox_mask.flatten()

        self.resx, self.resy, self.resz = resx, resy, resz

    @torch.compile
    def sample_valid_density_grid(self, frame):
        with torch.no_grad():
            input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * float(frame / 120.0)], dim=-1).reshape(-1, 4)
            raw_d_flat_list = []
            batch_size = 64 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d_valid = raw_d_flat[self.bbox_mask_flat]
            return raw_d_valid

    def plot_density_distribution_histogram(self, frame, bins):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            density_tensor = self.sample_valid_density_grid(frame)
            density_flat = density_tensor.view(-1).cpu().numpy()

            plt.figure(figsize=(8, 4))
            plt.hist(density_flat, bins=bins, edgecolor='black')
            plt.title("Density Distribution Histogram")
            plt.xlabel("Density Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            print(f"max density: {density_flat.max()}, min density: {density_flat.min()}, mean density: {density_flat.mean()}, std density: {density_flat.std()}")

            return density_flat
