from lib.evaluate import *
import lib.utils.houdini
import tqdm
import os
from phi.torch.flow import *


def open_file_dialog():
    from tkinter import filedialog, Tk
    while True:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(title="Select a checkpoint file", initialdir="history", filetypes=[("Checkpoint files", "*.tar")])
        root.destroy()

        if file_path and os.path.isfile(file_path):
            print("Successfully loaded checkpoint:", file_path)
            return file_path
        else:
            print("invalid file path, please select a valid checkpoint file.")


@jit_compile
def advect_step(v, s, dt):
    s = advect.mac_cormack(s, v, dt)
    return s


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    import argparse

    parser = argparse.ArgumentParser(description="Run validation.")
    parser.add_argument('--option', type=str, choices=[
        'evaluate_render_frame',
        'evaluate_render_grids',
        'evaluate_resimulation',
        'export_density_field',
        'export_velocity_field',
        'evaluate_density_details',
        'evaluate_lcc_bounding_box',
        'evaluate_diff_density',
        'evaluate_resimulation_test',
    ], required=True, help="[Required][General] Choose the operation to execute.")
    parser.add_argument('--device', type=str, default="cuda:0", help="[General] Device to run the operation.")
    parser.add_argument('--dtype', type=str, default="float32", choices=['float32', 'float16'], help="[General] Data type to use.")
    parser.add_argument('--select_ckpt', action='store_true', help="[General] Select a pretrained checkpoint file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="[General] Load a pretrained checkpoint.")
    parser.add_argument('--frame', type=int, default=-1, help="[General] Frame to evaluate.")
    parser.add_argument('--depth_size', type=int, default=256, help="[evaluate_render_frame only] Depth size for training.")
    parser.add_argument('--resx', type=int, default=128, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in x direction.")
    parser.add_argument('--resy', type=int, default=192, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in y direction.")
    parser.add_argument('--resz', type=int, default=128, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in z direction.")
    parser.add_argument('--batch_ray_size', type=int, default=1024 * 16, help="[evaluate_render_frame only] Batch ray size for training.")
    args = parser.parse_args()

    if args.select_ckpt and args.checkpoint is None:
        args.checkpoint = open_file_dialog()
    assert args.checkpoint is not None, "Checkpoint are required for evaluation."

    args_str = ' '.join([f'--{k}={v}' for k, v in vars(args).items()])
    print(f"\033[32m==================== Running command: {args_str} ====================\033[0m")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    checkpoint_config = checkpoint.get('config', {})
    scene_name = checkpoint_config.get('scene_name', 'unknown_scene')
    frame_start = int(checkpoint_config.get('frame_start', 0))
    frame_end = int(checkpoint_config.get('frame_end', 120))
    total_frames = frame_end - frame_start
    print(f"==================== Evaluating: {scene_name} ====================")
    print(f"Checkpoint Information: {checkpoint_config}, frame_start: {frame_start}, frame_end: {frame_end}")

    evaluation_config = EvaluationConfig(
        evaluation_script=args_str,
        scene_name=scene_name,
        pretrained_ckpt=args.checkpoint,
        target_device=torch.device(args.device),
        target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
        ratio=0.5,
        use_rgb=scene_name == "plume_color_1",
        frame_start=frame_start,
        frame_end=frame_end,
    )

    if args.option == "evaluate_render_frame":
        with torch.no_grad():
            model = EvaluationRenderFrame(evaluation_config)
            import imageio.v3 as imageio
            import imageio.v2 as imageiov2

            if args.frame == -1:
                rgb8_list = []
                for f in tqdm.tqdm(list(reversed(range(frame_start, frame_end))), desc="Rendering frames", unit="frame"):
                    rgb_map_final = model.render_frame(args.batch_ray_size, args.depth_size, frame_normalized=float(f) / float(total_frames))
                    rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                    os.makedirs(f'ckpt/{scene_name}/{model.tag}/render_frame', exist_ok=True)
                    imageio.imwrite(os.path.join(f'ckpt/{scene_name}/{model.tag}/render_frame', 'rgb_{:03d}.png'.format(f)), rgb8)
                    rgb8_list.append(rgb8)
                imageiov2.mimsave(os.path.join(f'ckpt/{scene_name}/{model.tag}/render_frame', 'video_rgb.mp4'.format(f)), list(reversed(rgb8_list)), fps=24)
            else:
                rgb_map_final = model.render_frame(args.batch_ray_size, args.depth_size, frame_normalized=float(args.frame) / float(total_frames))
                rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                os.makedirs(f'ckpt/{scene_name}/{model.tag}/render_frame', exist_ok=True)
                imageio.imwrite(os.path.join(f'ckpt/{scene_name}/{model.tag}/render_frame', 'rgb_{:03d}.png'.format(args.frame)), rgb8)

    if args.option == "evaluate_resimulation":
        with torch.no_grad():
            model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
            dt = 1.0 / float(total_frames - 1)
            source_height = 0.15
            den = model.sample_density_grid(frame_normalized=0.0)
            source = den
            for step in tqdm.trange(frame_start, frame_end):
                if step > frame_start:
                    vel = model.sample_velocity_grid(frame_normalized=float(step - 1) / float(total_frames))
                    vel_sim_confined = world2sim_rot(vel, model.s_w2s, model.s_scale)
                    source = model.sample_density_grid(frame_normalized=float(step) / float(total_frames))
                    den = model.advect_density(den, vel_sim_confined, source, dt, model.coord_3d_sim[..., 1] > source_height)
                lib.utils.houdini.export_density_field(
                    den,
                    save_path=f"ckpt/{scene_name}/{model.tag}/resimulation",
                    surname=f"density_{step + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )

    if args.option == 'evaluate_resimulation_test':
        with torch.no_grad():
            model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
            dt = 1.0 / float(total_frames - 1)
            den = model.sample_density_grid(frame_normalized=0.0)
            den_phi_tensor = math.tensor(den.squeeze(-1), spatial('x,y,z'))
            phi_smoke = CenteredGrid(den_phi_tensor, ZERO_GRADIENT, x=args.resx, y=args.resy, z=args.resz, bounds=Box(x=1, y=1, z=1))

            mask_to_sim = model.coord_3d_sim[..., 1] > 0.15
            for step in tqdm.trange(frame_start, frame_end):
                if step > frame_start:
                    vel = model.sample_velocity_grid(frame_normalized=float(step - 1) / float(total_frames))
                    vel_sim_confined = world2sim_rot(vel, model.s_w2s, model.s_scale)
                    phi_vel_tensor = math.tensor(vel_sim_confined, spatial('x,y,z'), channel('vector'))
                    phi_velocity = StaggeredGrid(phi_vel_tensor, 0, x=args.resx, y=args.resy, z=args.resz, bounds=Box(x=1, y=1, z=1))
                    phi_smoke = advect_step(phi_velocity, phi_smoke, dt)

                    # replace out of mask
                    source = model.sample_density_grid(frame_normalized=float(step) / float(total_frames)).squeeze(-1)
                    den_masked = phi_smoke.data.native('x,y,z')
                    den_masked[~mask_to_sim] = source[~mask_to_sim]
                    phi_smoke = CenteredGrid(math.tensor(den_masked, spatial('x,y,z')), ZERO_GRADIENT, x=args.resx, y=args.resy, z=args.resz, bounds=Box(x=1, y=1, z=1))
                lib.utils.houdini.export_density_field(
                    phi_smoke.data.native('x,y,z'),
                    save_path=f"ckpt/{scene_name}/{model.tag}/resimulation_phiflow",
                    surname=f"density_{step + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )

    if args.option == "export_density_field":
        resx_occupancy, resy_occupancy, resz_occupancy = 30, 30, 30
        model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
        model_occupancy = EvaluationDiscreteSpatial(evaluation_config, resx_occupancy, resy_occupancy, resz_occupancy)
        if args.frame == -1:
            for _ in tqdm.trange(frame_start, frame_end):
                lib.utils.houdini.export_density_field(
                    den=model.sample_density_grid(frame_normalized=float(_) / float(total_frames)),
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"density_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.create_voxel_boxes(model_occupancy.sample_density_grid(frame_normalized=float(_) / float(total_frames)) > 1e-5, f"ckpt/{scene_name}/{model.tag}/export", f"occupancy_grid_valid_{_ + 1:03d}", evaluation_config.s2w, evaluation_config.s_scale)
        else:
            lib.utils.houdini.export_density_field(
                den=model.sample_density_grid(frame_normalized=float(args.frame) / float(total_frames)),
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"density_{args.frame:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.create_voxel_boxes(model_occupancy.sample_density_grid(frame_normalized=float(args.frame) / float(total_frames)) > 1e-5, f"ckpt/{scene_name}/{model.tag}/export", f"occupancy_grid_valid_{args.frame:03d}", evaluation_config.s2w, evaluation_config.s_scale)

    if args.option == "export_velocity_field":
        model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
        if args.frame == -1:
            for _ in tqdm.trange(frame_start, frame_end):
                lib.utils.houdini.export_velocity_field(
                    vel=model.sample_velocity_grid(frame_normalized=float(_) / float(total_frames)),
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"velocity_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
        else:
            lib.utils.houdini.export_velocity_field(
                vel=model.sample_velocity_grid(frame_normalized=float(args.frame) / float(total_frames)),
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"velocity_{args.frame:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )

    if args.option == "evaluate_density_details":
        model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
        occupied_ratios = []
        for _ in tqdm.trange(frame_start, frame_end):
            den = model.sample_density_grid(frame_normalized=float(_) / float(total_frames))
            ratio = torch.sum(den > 1e-5).item() / (den.shape[0] * den.shape[1] * den.shape[2])
            occupied_ratios.append(ratio)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(range(frame_start, frame_end), occupied_ratios, marker='o')
        plt.xlabel("Frame")
        plt.ylabel("Occupied Ratio")
        plt.title(f"Scene: {scene_name} - Occupied Ratio over Frames")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if args.option == "evaluate_lcc_bounding_box":
        # Largest Connected Component (LCC) Bounding Box

        model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
        den_list = []
        bbox_min_list = []
        bbox_max_list = []
        for _ in tqdm.trange(frame_start, frame_end):
            bbox_min, bbox_max, den = model.max_component_bounding_box(frame_normalized=float(_) / float(total_frames), threshold=1e-5, extrapolate=5)
            den_list.append(den)
            bbox_min_list.append(bbox_min)
            bbox_max_list.append(bbox_max)


        def kalman_smooth_bbox(bbox_list):
            from pykalman import KalmanFilter
            arr = np.array(bbox_list)  # (N, 3)
            kf = KalmanFilter(initial_state_mean=arr[0],
                              n_dim_obs=3,
                              em_vars=['transition_covariance', 'observation_covariance'])
            kf = kf.em(arr, n_iter=10)
            smoothed, _ = kf.smooth(arr)
            return smoothed


        bbox_min_list_smoothed = kalman_smooth_bbox(bbox_min_list)
        bbox_max_list_smoothed = kalman_smooth_bbox(bbox_max_list)
        bbox_min_list_smoothed_floor = np.maximum(np.floor(bbox_min_list_smoothed), 0).astype(np.int32)
        bbox_max_list_smoothed_ceil = np.minimum(np.ceil(bbox_max_list_smoothed), np.array([args.resx - 1, args.resy - 1, args.resz - 1])).astype(np.int32)
        import yaml

        with open(f"{scene_name}_lcc.yaml", "w") as f:
            yaml.dump({
                'bbox_min_list_smoothed': bbox_min_list_smoothed.tolist(),
                'bbox_max_list_smoothed': bbox_max_list_smoothed.tolist(),
                'bbox_min_list_smoothed_floor': bbox_min_list_smoothed_floor.tolist(),
                'bbox_max_list_smoothed_ceil': bbox_max_list_smoothed_ceil.tolist(),
                'resx': args.resx,
                'resy': args.resy,
                'resz': args.resz,
            }, f, default_flow_style=None)

        sample_idx = 0
        for _ in tqdm.trange(frame_start, frame_end):
            lib.utils.houdini.export_density_field_with_bbox(
                den=den_list[sample_idx],
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"density_bbox_{_ + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
                bbox_min=bbox_min_list_smoothed_floor[sample_idx],
                bbox_max=bbox_max_list_smoothed_ceil[sample_idx],
            )
            sample_idx += 1

    if args.option == 'evaluate_diff_density':
        model = EvaluationDiscreteSpatial(evaluation_config, args.resx, args.resy, args.resz)
        if args.frame == -1:
            for _ in tqdm.trange(frame_start, frame_end):
                raw_d, d_x, d_y, d_z, d_t = model.sample_diff_density_grid(frame_normalized=float(_) / float(total_frames))
                print(f"raw_d: {raw_d.shape}, d_x: {d_x.shape}, d_y: {d_y.shape}, d_z: {d_z.shape}, d_t: {d_t.shape}")
                d_density = torch.cat([d_x, d_y, d_z], dim=-1)
                lib.utils.houdini.export_density_field(
                    den=raw_d,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"density_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.export_density_field(
                    den=d_x,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"d_x_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.export_density_field(
                    den=d_y,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"d_y_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.export_density_field(
                    den=d_z,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"d_z_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.export_density_field(
                    den=d_t,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"d_t_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.export_velocity_field(
                    vel=d_density,
                    save_path=f"ckpt/{scene_name}/{model.tag}/export",
                    surname=f"d_density_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
        else:
            raw_d, d_x, d_y, d_z, d_t = model.sample_diff_density_grid(frame_normalized=float(args.frame) / float(total_frames))
            print(f"raw_d: {raw_d.shape}, d_x: {d_x.shape}, d_y: {d_y.shape}, d_z: {d_z.shape}, d_t: {d_t.shape}")
            d_density = torch.cat([d_x, d_y, d_z], dim=-1)
            lib.utils.houdini.export_density_field(
                den=raw_d,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"density_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.export_density_field(
                den=d_x,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"d_x_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.export_density_field(
                den=d_y,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"d_y_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.export_density_field(
                den=d_z,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"d_z_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.export_density_field(
                den=d_t,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"d_t_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.export_velocity_field(
                vel=d_density,
                save_path=f"ckpt/{scene_name}/{model.tag}/export",
                surname=f"d_density_{args.frame + 1:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )

    print("\033[32m==================== Evaluation completed. ====================\033[0m")
