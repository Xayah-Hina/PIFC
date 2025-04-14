from train import *
from evaluate import *


def get_current_function_name():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


def train_density_only(config: TrainConfig, total_iter: int, pretrained_ckpt=None):
    model = TrainDensityModel(config)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{datetime.now().strftime('%m%d%H')}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        import tqdm
        for _ in tqdm.trange(total_iter):
            img_loss = model.forward(config.batch_size, config.depth_size)
            writer.add_scalar("Loss/img_loss", img_loss, _)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
        writer.close()


def train_velocity(config: TrainConfig, total_iter: int, pretrained_ckpt=None):
    model = TrainVelocityModel(config)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{datetime.now().strftime('%m%d%H')}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        import tqdm
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, proj_loss, min_vel_reg = model.forward(config.batch_size)
            writer.add_scalar("Loss/vel_loss", vel_loss, _)
            writer.add_scalar("Loss/nseloss_fine", nseloss_fine, _)
            writer.add_scalar("Loss/proj_loss", proj_loss, _)
            writer.add_scalar("Loss/min_vel_reg", min_vel_reg, _)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
        writer.close()


def train_joint(config: TrainConfig, total_iter: int, pretrained_ckpt=None):
    model = TrainJointModel(config)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{datetime.now().strftime('%m%d%H')}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        import tqdm
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, img_loss, proj_loss, min_vel_reg = model.forward(config.batch_size, config.depth_size)
            writer.add_scalar("Loss/vel_loss", vel_loss, _)
            writer.add_scalar("Loss/nseloss_fine", nseloss_fine, _)
            writer.add_scalar("Loss/img_loss", img_loss, _)
            writer.add_scalar("Loss/proj_loss", proj_loss, _)
            writer.add_scalar("Loss/min_vel_reg", min_vel_reg, _)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
        writer.close()


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


if __name__ == "__main__":
    print("==================== Operation starting. ====================")

    import argparse

    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['train_density_only', 'train_velocity', 'train_joint', 'evaluate_render_frame'], required=True, help="Choose the operation to execute.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the operation.")
    parser.add_argument('--dtype', type=str, default="float32", choices=['float32', 'float16'], help="Data type to use.")
    parser.add_argument('--scene', type=str, default="hyfluid", help="Scene to run.")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training.")
    parser.add_argument('--depth_size', type=int, default=128, help="Depth size for training.")
    parser.add_argument('--ratio', type=float, default=0.5, help="Ratio of resolution resampling.")
    parser.add_argument('--total_iter', type=int, default=10000, help="Total iterations for training.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load.")
    args = parser.parse_args()

    if args.option == "train_density_only":
        train_density_only(
            config=TrainConfig(
                scene_name=args.scene,
                target_device=torch.device(args.device),
                target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
                batch_size=args.batch_size,
                depth_size=args.depth_size,
                ratio=args.ratio,
            ),
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint
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
            ),
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint
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
            ),
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint
        )

    if args.option == "evaluate_render_frame":
        assert args.checkpoint is not None and args.scene is not None, "Checkpoint and scene name are required for evaluation."
        test_pose = torch.tensor([[0.4863, -0.2431, -0.8393, -0.7697],
                                  [-0.0189, 0.9574, -0.2882, 0.0132],
                                  [0.8736, 0.1560, 0.4610, 0.3250],
                                  [0.0000, 0.0000, 0.0000, 1.0000]], device=torch.device(args.device), dtype=torch.float32)
        test_focal = torch.tensor(2613.7634, device=torch.device(args.device), dtype=torch.float32)
        test_width = 1080
        test_height = 1920
        test_near = 1.1
        test_far = 1.5
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
            far=test_far
        )

    print("==================== Operation completed. ====================")
