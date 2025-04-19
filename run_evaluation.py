from lib.evaluate import *
import lib.utils.houdini
import tqdm
import os


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


if __name__ == "__main__":
    print("==================== Evaluation starting. ====================")
    torch.set_float32_matmul_precision('high')
    import argparse

    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['evaluate_render_frame', 'evaluate_resimulation', 'export_density_field', 'export_velocity_field'], required=True, help="[Required][General] Choose the operation to execute.")
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

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    scene_name = checkpoint['config']['scene_name']
    frame_start = int(checkpoint['config']['frame_start'])
    frame_end = int(checkpoint['config']['frame_end'])
    total_frames = frame_end - frame_start
    print(f"==================== Evaluating: {scene_name} ====================")
    print(f"Checkpoint Information: {checkpoint['config']}, frame_start: {frame_start}, frame_end: {frame_end}")

    evaluation_config = EvaluationConfig(
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
                    os.makedirs(f'ckpt/{scene_name}/render_frame', exist_ok=True)
                    imageio.imwrite(os.path.join(f'ckpt/{scene_name}/render_frame', 'rgb_{:03d}.png'.format(f)), rgb8)
                    rgb8_list.append(rgb8)
                imageiov2.mimsave(os.path.join(f'ckpt/{scene_name}/render_frame', 'video_rgb.mp4'.format(f)), list(reversed(rgb8_list)), fps=24)
            else:
                rgb_map_final = model.render_frame(args.batch_ray_size, args.depth_size, frame_normalized=float(args.frame) / float(total_frames))
                rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                os.makedirs(f'ckpt/{scene_name}/render_frame', exist_ok=True)
                imageio.imwrite(os.path.join(f'ckpt/{scene_name}/render_frame', 'rgb_{:03d}.png'.format(args.frame)), rgb8)

    if args.option == "evaluate_resimulation":
        with torch.no_grad():
            model = EvaluationResimulation(evaluation_config, args.resx, args.resy, args.resz)
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
                    save_path=f"ckpt/{scene_name}/resimulation",
                    surname=f"density_{step:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )

    if args.option == "export_density_field":
        resx_occupancy, resy_occupancy, resz_occupancy = 30, 30, 30
        model = EvaluationResimulation(evaluation_config, args.resx, args.resy, args.resz)
        model_occupancy = EvaluationResimulation(evaluation_config, resx_occupancy, resy_occupancy, resz_occupancy)
        if args.frame == -1:
            for _ in tqdm.trange(frame_start, frame_end):
                lib.utils.houdini.export_density_field(
                    den=model.sample_density_grid(frame_normalized=float(_) / float(total_frames)),
                    save_path=f"ckpt/{scene_name}/export",
                    surname=f"density_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
                lib.utils.houdini.create_voxel_boxes(model_occupancy.sample_density_grid(frame_normalized=float(_) / float(total_frames)) > 1e-5, f"ckpt/{scene_name}/export", f"occupancy_grid_valid_{_ + 1:03d}", evaluation_config.s2w, evaluation_config.s_scale)
        else:
            lib.utils.houdini.export_density_field(
                den=model.sample_density_grid(frame_normalized=float(args.frame) / float(total_frames)),
                save_path=f"ckpt/{scene_name}/export",
                surname=f"density_{args.frame:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )
            lib.utils.houdini.create_voxel_boxes(model_occupancy.sample_density_grid(frame_normalized=float(args.frame) / float(total_frames)) > 1e-5, f"ckpt/{scene_name}/export", f"occupancy_grid_valid_{args.frame:03d}", evaluation_config.s2w, evaluation_config.s_scale)

    if args.option == "export_velocity_field":
        model = EvaluationResimulation(evaluation_config, args.resx, args.resy, args.resz)
        if args.frame == -1:
            for _ in tqdm.trange(frame_start, frame_end):
                lib.utils.houdini.export_velocity_field(
                    vel=model.sample_velocity_grid(frame_normalized=float(_) / float(total_frames)),
                    save_path=f"ckpt/{scene_name}/export",
                    surname=f"velocity_{_ + 1:03d}",
                    local2world=model.s2w,
                    scale=model.s_scale,
                )
        else:
            lib.utils.houdini.export_velocity_field(
                vel=model.sample_velocity_grid(frame_normalized=float(args.frame) / float(total_frames)),
                save_path=f"ckpt/{scene_name}/export",
                surname=f"velocity_{args.frame:03d}",
                local2world=model.s2w,
                scale=model.s_scale,
            )

    print("==================== Evaluation completed. ====================")
