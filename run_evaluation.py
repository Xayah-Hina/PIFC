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
        file_path = filedialog.askopenfilename(title="Select a checkpoint file", initialdir="ckpt", filetypes=[("Checkpoint files", "*.tar")])
        root.destroy()

        if file_path and os.path.isfile(file_path):
            print("Successfully loaded checkpoint:", file_path)
            return file_path
        else:
            print("invalid file path, please select a valid checkpoint file.")


if __name__ == "__main__":
    print("==================== Evaluation starting. ====================")

    import argparse

    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['evaluate_render_frame', 'evaluate_resimulation', 'export_density_field', 'export_velocity_field'], required=True, help="[Required][General] Choose the operation to execute.")
    parser.add_argument('--device', type=str, default="cuda:0", help="[General] Device to run the operation.")
    parser.add_argument('--dtype', type=str, default="float32", choices=['float32', 'float16'], help="[General] Data type to use.")
    parser.add_argument('--select_ckpt', action='store_true', help="[General] Select a pretrained checkpoint file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="[General] Load a pretrained checkpoint.")
    parser.add_argument('--frame', type=int, default=-1, help="[General] Frame to evaluate.")
    parser.add_argument('--depth_size', type=int, default=128, help="[evaluate_render_frame only] Depth size for training.")
    parser.add_argument('--resx', type=int, default=128, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in x direction.")
    parser.add_argument('--resy', type=int, default=192, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in y direction.")
    parser.add_argument('--resz', type=int, default=128, help="[evaluate_resimulation, export_density_field, export_velocity_field] Resolution in z direction.")
    parser.add_argument('--batch_ray_size', type=int, default=1024 * 32, help="[evaluate_render_frame only] Batch ray size for training.")
    args = parser.parse_args()

    if args.select_ckpt and args.checkpoint is None:
        args.checkpoint = open_file_dialog()
    assert args.checkpoint is not None, "Checkpoint are required for evaluation."

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    scene_name = checkpoint['config']['scene_name']
    print(f"==================== Evaluating: {scene_name} ====================")
    print(f"Checkpoint Information: {checkpoint['config']}")

    config = EvaluationConfig(
        scene_name=scene_name,
        pretrained_ckpt=args.checkpoint,
        target_device=torch.device(args.device),
        target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
    )

    if args.option == "evaluate_render_frame":
        if scene_name == "hyfluid":
            pose = torch.tensor([[0.4863, -0.2431, -0.8393, -0.7697],
                                 [-0.0189, 0.9574, -0.2882, 0.0132],
                                 [0.8736, 0.1560, 0.4610, 0.3250],
                                 [0.0000, 0.0000, 0.0000, 1.0000]], device=torch.device(args.device), dtype=torch.float32)
            focal = torch.tensor(2613.7634, device=torch.device(args.device), dtype=torch.float32)
            width = 1080
            height = 1920
            near = 1.1
            far = 1.5
        else:
            pose = torch.tensor([[-6.5174e-01, 7.3241e-02, 7.5490e-01, 3.5361e+00],
                                 [-6.9389e-18, 9.9533e-01, -9.6567e-02, 1.9000e+00],
                                 [-7.5844e-01, -6.2937e-02, -6.4869e-01, -2.6511e+00],
                                 [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], device=torch.device(args.device), dtype=torch.float32)
            focal = torch.tensor(1303.6753, device=torch.device(args.device), dtype=torch.float32)
            width = 1080
            height = 1920
            near = 2.5
            far = 5.4

        ratio = 1.0
        focal = focal * ratio
        width = int(width * ratio)
        height = int(height * ratio)

        with torch.no_grad():
            model = EvaluationRenderFrame(config)
            import imageio.v3 as imageio

            if args.frame == -1:
                for f in tqdm.tqdm(list(reversed(range(120))), desc="Rendering frames", unit="frame"):
                    rgb_map_final = model.render_frame(args.batch_ray_size, pose, focal, width, height, args.depth_size, near, far, f)
                    rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                    os.makedirs(f'ckpt/{scene_name}/render_frame', exist_ok=True)
                    imageio.imwrite(os.path.join(f'ckpt/{scene_name}/render_frame', 'rgb_{:03d}.png'.format(f)), rgb8)
            else:
                rgb_map_final = model.render_frame(args.batch_ray_size, pose, focal, width, height, args.depth_size, near, far, args.frame)
                rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                os.makedirs(f'ckpt/{scene_name}/render_frame', exist_ok=True)
                imageio.imwrite(os.path.join(f'ckpt/{scene_name}/render_frame', 'rgb_{:03d}.png'.format(args.frame)), rgb8)

    if args.option == "evaluate_resimulation":
        with torch.no_grad():
            model = EvaluationResimulation(config, args.resx, args.resy, args.resz)
            dt = 1.0 / 119.0
            source_height = 0.15
            den = model.sample_density_grid(0)
            source = den
            for step in tqdm.trange(120):
                if step > 0:
                    vel = model.sample_velocity_grid(step - 1)
                    vel_sim_confined = world2sim_rot(vel, model.s_w2s, model.s_scale)
                    source = model.sample_density_grid(step)
                    den = model.advect_density(den, vel_sim_confined, source, dt, model.coord_3d_sim[..., 1] > source_height)
                lib.utils.houdini.export_density_field(
                    den,
                    save_path="ckpt/resimulation",
                    surname=f"density_{step:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )

    if args.option == "export_density_field":
        model = EvaluationResimulation(config, args.resx, args.resy, args.resz)
        if args.frame == -1:
            for _ in tqdm.trange(120):
                lib.utils.houdini.export_density_field(
                    den=model.sample_density_grid(frame=_ + 1),
                    save_path="ckpt/export",
                    surname=f"density_{_ + 1:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )
        else:
            lib.utils.houdini.export_density_field(
                den=model.sample_density_grid(frame=args.frame),
                save_path="ckpt/export",
                surname=f"density_{args.frame:03d}",
                bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
            )

    if args.option == "export_velocity_field":
        model = EvaluationResimulation(config, args.resx, args.resy, args.resz)
        if args.frame == -1:
            for _ in tqdm.trange(120):
                lib.utils.houdini.export_velocity_field(
                    vel=model.sample_velocity_grid(frame=_ + 1),
                    save_path="ckpt/export",
                    surname=f"velocity_{_ + 1:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )
        else:
            lib.utils.houdini.export_velocity_field(
                vel=model.sample_velocity_grid(frame=frame),
                save_path="ckpt/export",
                surname=f"velocity_{frame:03d}",
                bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
            )

    print("==================== Evaluation completed. ====================")
