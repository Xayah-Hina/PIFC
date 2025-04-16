from train import *
from evaluate import *


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


def get_current_function_name():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


def train_density_only(config: TrainConfig, total_iter: int, pretrained_ckpt=None):
    model = TrainDensityModel(config)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    date = datetime.now().strftime('%m%d%H%M%S')
    device_str = f"{config.target_device.type}{config.target_device.index if config.target_device.index is not None else ''}"
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{date}/{device_str}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        import tqdm
        for _ in tqdm.trange(total_iter):
            img_loss = model.forward(config.batch_size, config.depth_size)
            writer.add_scalar(f"Loss/{date}/{device_str}/img_loss", img_loss, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.use_mid_ckpts and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


def train_velocity(config: TrainConfig, pretrain_density: int, resx: int, resy: int, resz: int, total_iter: int, pretrained_ckpt=None):
    model = TrainVelocityModel(config, resx, resy, resz)
    if pretrained_ckpt is None:
        pretrained_ckpt = train_density_only(config, pretrain_density, pretrained_ckpt=None)
    model.load_ckpt(pretrained_ckpt, config.target_device)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    date = datetime.now().strftime('%m%d%H%M%S')
    device_str = f"{config.target_device.type}{config.target_device.index if config.target_device.index is not None else ''}"
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{date}/{device_str}")
    try:
        import tqdm
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, proj_loss, min_vel_reg = model.forward(config.batch_size)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.use_mid_ckpts and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


def train_joint(config: TrainConfig, total_iter: int, pretrained_ckpt=None):
    model = TrainJointModel(config)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    date = datetime.now().strftime('%m%d%H%M%S')
    device_str = f"{config.target_device.type}{config.target_device.index if config.target_device.index is not None else ''}"
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{date}/{device_str}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        import tqdm
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, img_loss, proj_loss, min_vel_reg = model.forward(config.batch_size, config.depth_size)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/img_loss", img_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.use_mid_ckpts and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


def evaluate_render_frame(config: EvaluationConfig, frame, pose, focal, width, height, depth_size, near, far):
    with torch.no_grad():
        model = EvaluationRenderFrame(config)
        if isinstance(frame, int):
            rgb_map_final = model.render_frame(pose, focal, width, height, depth_size, near, far, frame)
            rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
            import imageio.v3 as imageio
            os.makedirs(f'ckpt/{config.scene_name}/render_frame', exist_ok=True)
            imageio.imwrite(os.path.join(f'ckpt/{config.scene_name}/render_frame', 'rgb_{:03d}.png'.format(frame)), rgb8)
        elif isinstance(frame, list):
            import tqdm
            for f in tqdm.tqdm(frame, desc="Rendering frames", unit="frame"):
                rgb_map_final = model.render_frame(pose, focal, width, height, depth_size, near, far, f)
                rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
                import imageio.v3 as imageio
                os.makedirs(f'ckpt/{config.scene_name}/render_frame', exist_ok=True)
                imageio.imwrite(os.path.join(f'ckpt/{config.scene_name}/render_frame', 'rgb_{:03d}.png'.format(f)), rgb8)
        else:
            raise ValueError("frame should be an integer or a list of integers.")


def evaluate_resimulation(config: EvaluationConfig, resx: int, resy: int, resz: int):
    with torch.no_grad():
        model = EvaluationResimulation(config, resx, resy, resz)
        dt = 1.0 / 119.0
        source_height = 0.15
        den = model.sample_density_grid(0)
        source = den

        import tqdm
        for step in tqdm.trange(120):
            if step > 0:
                vel = model.sample_velocity_grid(step - 1)
                vel_sim_confined = world2sim_rot(vel, model.s_w2s, model.s_scale)
                source = model.sample_density_grid(step)
                den = model.advect_density(den, vel_sim_confined, source, dt, model.coord_3d_sim[..., 1] > source_height)
            os.makedirs(f'ckpt/{config.scene_name}/resimulation', exist_ok=True)
            np.save(os.path.join(f'ckpt/{config.scene_name}/resimulation', f'density_advected_{step + 1:03d}.npy'), den.cpu().numpy())
            np.save(os.path.join(f'ckpt/{config.scene_name}/resimulation', f'density_original_{step + 1:03d}.npy'), source.cpu().numpy())


if __name__ == "__main__":
    print("==================== Operation starting. ====================")

    import argparse

    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['train_density_only', 'train_velocity', 'train_joint', 'evaluate_render_frame', 'evaluate_resimulation', 'export_density_field', 'export_velocity_field'], required=True, help="Choose the operation to execute.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the operation.")
    parser.add_argument('--dtype', type=str, default="float32", choices=['float32', 'float16'], help="Data type to use.")
    parser.add_argument('--scene', type=str, default="hyfluid", help="Scene to run.")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training.")
    parser.add_argument('--depth_size', type=int, default=128, help="Depth size for training.")
    parser.add_argument('--ratio', type=float, default=0.5, help="Ratio of resolution resampling.")
    parser.add_argument('--total_iter', type=int, default=10000, help="Total iterations for training.")
    parser.add_argument('--select_ckpt', action='store_true', help="Select a checkpoint file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load.")
    parser.add_argument('--resx', type=int, default=128, help="Resolution in x direction.")
    parser.add_argument('--resy', type=int, default=192, help="Resolution in y direction.")
    parser.add_argument('--resz', type=int, default=128, help="Resolution in z direction.")
    parser.add_argument('--frame', type=int, default=-1, help="Frame to render.")
    parser.add_argument('--use_mid_ckpts', type=bool, default=False, help="Use mid-ckpts iteration.")
    parser.add_argument('--mid_ckpts_iters', type=int, default=2000, help="Mid-ckpts iteration.")
    args = parser.parse_args()

    if args.select_ckpt and args.checkpoint is None:
        args.checkpoint = open_file_dialog()

    if args.option == "train_density_only":
        train_density_only(
            config=TrainConfig(
                scene_name=args.scene,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
                batch_size=args.batch_size,
                depth_size=args.depth_size,
                ratio=args.ratio,
                use_mid_ckpts=args.use_mid_ckpts,
                mid_ckpts_iters=args.mid_ckpts_iters,
            ),
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_velocity":
        train_velocity(
            config=TrainConfig(
                scene_name=args.scene,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
                batch_size=args.batch_size,
                depth_size=args.depth_size,
                ratio=args.ratio,
                use_mid_ckpts=args.use_mid_ckpts,
                mid_ckpts_iters=args.mid_ckpts_iters,
            ),
            pretrain_density=100000,
            resx=args.resx,
            resy=args.resy,
            resz=args.resz,
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_joint":
        train_joint(
            config=TrainConfig(
                scene_name=args.scene,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
                batch_size=args.batch_size,
                depth_size=args.depth_size,
                ratio=args.ratio,
                use_mid_ckpts=args.use_mid_ckpts,
                mid_ckpts_iters=args.mid_ckpts_iters,
            ),
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "evaluate_render_frame":
        assert args.checkpoint is not None and args.scene is not None, "Checkpoint and scene name are required for evaluation."
        if args.scene == "hyfluid":
            test_pose = torch.tensor([[0.4863, -0.2431, -0.8393, -0.7697],
                                      [-0.0189, 0.9574, -0.2882, 0.0132],
                                      [0.8736, 0.1560, 0.4610, 0.3250],
                                      [0.0000, 0.0000, 0.0000, 1.0000]], device=torch.device(args.device), dtype=torch.float32)
            test_focal = torch.tensor(2613.7634, device=torch.device(args.device), dtype=torch.float32)
            test_width = 1080
            test_height = 1920
            test_near = 1.1
            test_far = 1.5
        else:
            test_pose = torch.tensor([[-6.5174e-01, 7.3241e-02, 7.5490e-01, 3.5361e+00],
                                      [-6.9389e-18, 9.9533e-01, -9.6567e-02, 1.9000e+00],
                                      [-7.5844e-01, -6.2937e-02, -6.4869e-01, -2.6511e+00],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], device=torch.device(args.device), dtype=torch.float32)
            test_focal = torch.tensor(1303.6753, device=torch.device(args.device), dtype=torch.float32)
            test_width = 1080
            test_height = 1920
            test_near = 2.5
            test_far = 5.4

        ratio = 1.0
        test_focal = test_focal * ratio
        test_width = int(test_width * ratio)
        test_height = int(test_height * ratio)

        evaluate_render_frame(
            config=EvaluationConfig(
                scene_name=args.scene,
                pretrained_ckpt=args.checkpoint,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
            ),
            frame=list(reversed(range(120))),
            pose=test_pose,
            focal=test_focal,
            width=test_width,
            height=test_height,
            depth_size=args.depth_size,
            near=test_near,
            far=test_far,
        )

    ##### Houdini Evaluation #####
    import lib.houdini
    import tqdm

    if args.option == "evaluate_resimulation":
        assert args.checkpoint is not None and args.scene is not None, "Checkpoint and scene name are required for evaluation."
        with torch.no_grad():
            model = EvaluationResimulation(
                config=EvaluationConfig(
                    scene_name=args.scene,
                    pretrained_ckpt=args.checkpoint,
                    target_device=torch.device(args.device),
                    target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
                ),
                resx=args.resx,
                resy=args.resy,
                resz=args.resz,
            )
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
                lib.houdini.export_density_field(
                    den,
                    save_path="ckpt/resimulation",
                    surname=f"density_{step:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )

    if args.option == "export_density_field":
        assert args.checkpoint is not None and args.scene is not None, "Checkpoint and scene name are required for evaluation."
        model = EvaluationResimulation(
            config=EvaluationConfig(
                scene_name=args.scene,
                pretrained_ckpt=args.checkpoint,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
            ),
            resx=args.resx,
            resy=args.resy,
            resz=args.resz,
        )
        frame = args.frame
        if frame == -1:
            for _ in tqdm.trange(120):
                lib.houdini.export_density_field(
                    den=model.sample_density_grid(frame=_ + 1),
                    save_path="ckpt/export",
                    surname=f"density_{_ + 1:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )
        else:
            lib.houdini.export_density_field(
                den=model.sample_density_grid(frame=frame),
                save_path="ckpt/export",
                surname=f"density_{frame:03d}",
                bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
            )

    if args.option == "export_velocity_field":
        model = EvaluationResimulation(
            config=EvaluationConfig(
                scene_name=args.scene,
                pretrained_ckpt=args.checkpoint,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
            ),
            resx=args.resx,
            resy=args.resy,
            resz=args.resz,
        )
        frame = args.frame
        if frame == -1:
            for _ in tqdm.trange(120):
                lib.houdini.export_velocity_field(
                    vel=model.sample_velocity_grid(frame=_ + 1),
                    save_path="ckpt/export",
                    surname=f"velocity_{_ + 1:03d}",
                    bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
                )
        else:
            lib.houdini.export_velocity_field(
                vel=model.sample_velocity_grid(frame=frame),
                save_path="ckpt/export",
                surname=f"velocity_{frame:03d}",
                bbox=(0.0, 0.0, 0.0, model.s_scale[0].item(), model.s_scale[1].item(), model.s_scale[2].item()),
            )

    print("==================== Operation completed. ====================")
