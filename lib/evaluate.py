from .utils.dataset import *
from .utils.frustum import *
from .utils.solver import *
from .model.encoder_hyfluid import *
from .model.model_hyfluid import *
import torch
import dataclasses


@dataclasses.dataclass
class EvaluationConfig:
    # evaluation script
    evaluation_script: str

    # checkpoint path
    pretrained_ckpt: str
    scene_name: str

    # general parameters
    target_device: torch.device
    target_dtype: torch.dtype

    ratio: float

    use_rgb: bool

    frame_start: int
    frame_end: int

    background_color: torch.Tensor = None

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
        self.tag = 'DEFAULT_TAG'
        self._load_model(config.target_device, config.target_dtype, config.use_rgb)
        self._load_validation_dataset(config.validation_videos, config.validation_camera_calibrations, config.frame_start, config.frame_end, config.ratio, config.target_device, config.target_dtype)
        self.load_ckpt(config.pretrained_ckpt, config.target_device)

        self.target_device = config.target_device
        self.target_dtype = config.target_dtype
        self.s_w2s, self.s2w, self.s_scale, self.s_min, self.s_max = config.s_w2s, config.s2w, config.s_scale, config.s_min, config.s_max
        self.ratio = config.ratio

        self.background_color = config.background_color

    def _load_model(self, target_device: torch.device, target_dtype: torch.dtype, use_rgb):
        self.encoder_d = HashEncoderNativeFasterBackward(device=target_device, dtype=target_dtype).to(target_device)
        if use_rgb:
            self.model_d = NeRFSmall_c(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2, dtype=target_dtype).to(target_device)
        else:
            self.model_d = NeRFSmall(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_d.num_levels * 2, dtype=target_dtype).to(target_device)
        self.encoder_v = HashEncoderNativeFasterBackward(device=target_device, dtype=target_dtype).to(target_device)
        self.model_v = NeRFSmallPotential(num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=2, hidden_dim_color=16, input_ch=self.encoder_v.num_levels * 2, use_f=False, dtype=target_dtype).to(target_device)

    def load_ckpt(self, path: str, device: torch.device):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            self.model_d.load_state_dict(checkpoint['model_d'])
            self.encoder_d.load_state_dict(checkpoint['encoder_d'])
            self.model_v.load_state_dict(checkpoint['model_v'])
            self.encoder_v.load_state_dict(checkpoint['encoder_v'])
            self.tag = checkpoint.get('config', {}).get('train_tag', None)
            print(f"loaded checkpoint from {path}, TAG: {self.tag}")
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
    def render_frame(self, batch_ray_size: int, depth_size: int, frame_normalized: float):
        pose = self.poses_validation[0]
        focal = self.focals_validation[0]
        with torch.no_grad():
            poses = pose.unsqueeze(0)
            focals = focal.unsqueeze(0)
            frame_time = torch.tensor(frame_normalized, device=self.target_device, dtype=self.target_dtype)

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

                if self.background_color:
                    acc_map = torch.sum(weights, -1)
                    rgb_map = rgb_map + self.background_color * (1.0 - acc_map[..., None])

                final_rgb_map_list.append(rgb_map)

            final_rgb_map = torch.cat(final_rgb_map_list, dim=0).reshape(self.height, self.width, 3)
            return final_rgb_map


class EvaluationDiscreteSpatial(_EvaluationModelBase):
    def __init__(self, config: EvaluationConfig, resx: int, resy: int, resz: int):
        super().__init__(config)

        xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, resx, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resy, device=self.target_device, dtype=self.target_dtype), torch.linspace(0, 1, resz, device=self.target_device, dtype=self.target_dtype)], indexing='ij')
        self.coord_3d_sim = torch.stack([xs, ys, zs], dim=-1)
        self.coord_3d_world = sim2world(self.coord_3d_sim, self.s2w, self.s_scale)
        self.bbox_mask = insideMask(self.coord_3d_world, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False)
        self.bbox_mask_flat = self.bbox_mask.flatten()

        self.resx, self.resy, self.resz = resx, resy, resz

    # @torch.compile
    def render_frame_grid(self, density_grid, batch_ray_size: int, depth_size: int):
        pose = self.poses_validation[0]
        focal = self.focals_validation[0]
        with torch.no_grad():
            poses = pose.unsqueeze(0)
            focals = focal.unsqueeze(0)

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

                bbox_mask = insideMask(batch_points, self.s_w2s, self.s_scale, self.s_min, self.s_max, to_float=False).flatten()

                batch_points_sim = world2sim(batch_points, self.s_w2s, self.s_scale)
                batch_points_sim_sample = batch_points_sim * 2 - 1  # [batch_size_current, N_depths, 3]
                batch_points_sim_sample_flat = batch_points_sim_sample.reshape(-1, 3)
                batch_points_sim_sample_flat_filtered = batch_points_sim_sample_flat[bbox_mask]

                vol = density_grid[None, ...].permute([0, 4, 3, 2, 1])  # [1, 1, resz, resy, resx]
                grid = batch_points_sim_sample_flat_filtered[None, ..., None, None, :]  # [1, BATCH * DEPTH, 1, 1, 3]
                den_sampled = torch.nn.functional.grid_sample(vol, grid, align_corners=True)

                raw_flat = torch.zeros([batch_size_current * depth_size], device=self.target_device, dtype=self.target_dtype)
                raw_d_flat = den_sampled.flatten()
                raw_flat = raw_flat.masked_scatter(bbox_mask, raw_d_flat)
                raw = raw_flat.reshape(batch_size_current, depth_size, 1)  # [batch_size_current, N_depths, 1]

                dists_cat = torch.cat([batch_dist_vals, torch.tensor([1e10], device=self.target_device).expand(batch_dist_vals[..., :1].shape)], -1)  # [batch_size_current, N_depths]
                dists_final = dists_cat * torch.norm(batch_rays_d[..., None, :], dim=-1)  # [batch_size_current, N_depths]

                noise = 0.
                alpha = 1. - torch.exp(-torch.nn.functional.relu(raw[..., -1] + noise) * dists_final)
                weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.target_device), 1. - alpha + 1e-10], -1), -1)[:, :-1]

                # TODO: support color
                rgb_trained = torch.ones(3, device=self.target_device) * (0.6 + torch.tanh(self.model_d.rgb) * 0.4)
                rgb_map = torch.sum(weights[..., None] * rgb_trained, -2)

                final_rgb_map_list.append(rgb_map)

            final_rgb_map = torch.cat(final_rgb_map_list, dim=0).reshape(self.height, self.width, 3)
            return final_rgb_map

    @torch.compile
    def sample_density_grid(self, frame_normalized):
        with torch.no_grad():
            input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * frame_normalized], dim=-1).reshape(-1, 4)
            raw_d_flat_list = []
            batch_size = 32 * 64 * 64
            for i in range(0, input_xyzt_flat.shape[0], batch_size):
                input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
                raw_d_flat_batch = self.model_d(self.encoder_d(input_xyzt_flat_batch))[..., 0]
                raw_d_flat_list.append(raw_d_flat_batch)
            raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
            raw_d_flat[~self.bbox_mask_flat] = 0.0
            raw_d = raw_d_flat.reshape(self.resx, self.resy, self.resz, 1)
            return raw_d

    def sample_diff_density_grid(self, frame_normalized):
        def g(x):
            return self.model_d(x)

        input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * frame_normalized], dim=-1).reshape(-1, 4)
        input_xyzt_flat.requires_grad = True

        raw_d_flat_list, d_x_list, d_y_list, d_z_list, d_t_list = [], [], [], [], []
        batch_size = 32 * 64 * 64
        for i in range(0, input_xyzt_flat.shape[0], batch_size):
            input_xyzt_flat_batch = input_xyzt_flat[i:i + batch_size]
            hidden = self.encoder_d(input_xyzt_flat_batch)
            raw_d_flat_batch = self.model_d(hidden)[..., 0]

            jac = torch.vmap(torch.func.jacrev(g))(hidden)
            jac_x = get_minibatch_jacobian(hidden, input_xyzt_flat_batch)
            jac = jac @ jac_x
            _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)]
            d_x_list.append(_d_x.detach().clone())
            d_y_list.append(_d_y.detach().clone())
            d_z_list.append(_d_z.detach().clone())
            d_t_list.append(_d_t.detach().clone())

            raw_d_flat_list.append(raw_d_flat_batch.detach().clone())
        raw_d_flat = torch.cat(raw_d_flat_list, dim=0)
        d_x_flat = torch.cat(d_x_list, dim=0)
        d_y_flat = torch.cat(d_y_list, dim=0)
        d_z_flat = torch.cat(d_z_list, dim=0)
        d_t_flat = torch.cat(d_t_list, dim=0)

        # raw_d_flat[~self.bbox_mask_flat] = 0.0 # no need for filter
        raw_d = raw_d_flat.reshape(self.resx, self.resy, self.resz, 1)
        d_x = d_x_flat.reshape(self.resx, self.resy, self.resz, 1)
        d_y = d_y_flat.reshape(self.resx, self.resy, self.resz, 1)
        d_z = d_z_flat.reshape(self.resx, self.resy, self.resz, 1)
        d_t = d_t_flat.reshape(self.resx, self.resy, self.resz, 1)
        return raw_d, d_x, d_y, d_z, d_t

    @torch.compile
    def sample_velocity_grid(self, frame_normalized):
        with torch.no_grad():
            input_xyzt_flat = torch.cat([self.coord_3d_world, torch.ones_like(self.coord_3d_world[..., :1]) * frame_normalized], dim=-1).reshape(-1, 4)
            raw_vel_flat_list = []
            batch_size = 32 * 64 * 64
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

    def max_component_bounding_box(self, frame_normalized, threshold, extrapolate):
        den = self.sample_density_grid(frame_normalized)
        occupancy_mask = (den > threshold).squeeze(-1).cpu().numpy()
        from scipy.ndimage import label, find_objects
        structure = np.ones((3, 3, 3))
        labeled, num_features = label(occupancy_mask, structure=structure)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # remove background
        max_label = sizes.argmax()
        main_region = (labeled == max_label)
        slices = find_objects(main_region.astype(int))[0]

        z_min, z_max = slices[2].start, slices[2].stop
        y_min, y_max = slices[1].start, slices[1].stop
        x_min, x_max = slices[0].start, slices[0].stop

        z_min, z_max = max(0, z_min - extrapolate), min(self.resz - 1, z_max + extrapolate)
        y_min, y_max = max(0, y_min - extrapolate), min(self.resy - 1, y_max + extrapolate)
        x_min, x_max = max(0, x_min - extrapolate), min(self.resx - 1, x_max + extrapolate)

        bbox_min = [x_min, y_min, z_min]
        bbox_max = [x_max, y_max, z_max]

        return bbox_min, bbox_max, den
