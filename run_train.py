from lib.train import *
import tqdm


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
        for _ in tqdm.trange(total_iter):
            img_loss = model.forward(config.batch_size, config.depth_size)
            writer.add_scalar(f"Loss/{date}/{device_str}/img_loss", img_loss, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            # if _ % 1000 == 0 and _ > 0:
            #     validation_loss = model.validate(config.depth_size)
            #     writer.add_scalar(f"Validation/{date}/{device_str}/validation_loss", validation_loss, _)

            if config.mid_ckpts_iters != -1 and model.global_step % config.mid_ckpts_iters == 0:
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
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, proj_loss, min_vel_reg = model.forward(config.batch_size, config.loss_dict)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.mid_ckpts_iters != -1 and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


def train_velocity_lcc(config: TrainConfig, lcc_path: str, pretrain_density: int, total_iter: int, pretrained_ckpt=None):
    model = TrainVelocityLCCModel(config, lcc_path)
    if pretrained_ckpt is None:
        pretrained_ckpt = train_density_only(config, pretrain_density, pretrained_ckpt=None)
    model.load_ckpt(pretrained_ckpt, config.target_device)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    date = datetime.now().strftime('%m%d%H%M%S')
    device_str = f"{config.target_device.type}{config.target_device.index if config.target_device.index is not None else ''}"
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{date}/{device_str}")
    try:
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, proj_loss, min_vel_reg, lcc_loss = model.forward(config.batch_size, config.loss_dict)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            if lcc_loss is not None:
                writer.add_scalar(f"Loss/{date}/{device_str}/lcc_loss", lcc_loss, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.mid_ckpts_iters != -1 and model.global_step % config.mid_ckpts_iters == 0:
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
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, img_loss, proj_loss, min_vel_reg = model.forward(config.batch_size, config.depth_size, config.loss_dict)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/img_loss", img_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.mid_ckpts_iters != -1 and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


def train_joint_lcc(config: TrainConfig, lcc_path: str, total_iter: int, pretrained_ckpt=None):
    model = TrainJointLCCModel(config, lcc_path)
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    date = datetime.now().strftime('%m%d%H%M%S')
    device_str = f"{config.target_device.type}{config.target_device.index if config.target_device.index is not None else ''}"
    writer = SummaryWriter(log_dir=f"ckpt/tensorboard/{get_current_function_name()}/{date}/{device_str}")
    try:
        if pretrained_ckpt:
            model.load_ckpt(pretrained_ckpt, config.target_device)
        for _ in tqdm.trange(total_iter):
            vel_loss, nseloss_fine, img_loss, proj_loss, min_vel_reg, lcc_loss = model.forward(config.batch_size, config.depth_size, config.loss_dict)
            writer.add_scalar(f"Loss/{date}/{device_str}/vel_loss", vel_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/nseloss_fine", nseloss_fine, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/img_loss", img_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/proj_loss", proj_loss, _)
            writer.add_scalar(f"Loss/{date}/{device_str}/min_vel_reg", min_vel_reg, _)
            if lcc_loss is not None:
                writer.add_scalar(f"Loss/{date}/{device_str}/lcc_loss", lcc_loss, _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_d", model.scheduler_d.get_last_lr()[0], _)
            writer.add_scalar(f"LearningRate/{date}/{device_str}/scheduler_v", model.scheduler_v.get_last_lr()[0], _)

            if config.mid_ckpts_iters != -1 and model.global_step % config.mid_ckpts_iters == 0:
                model.save_ckpt(f'ckpt/{config.scene_name}/{get_current_function_name()}', final=False)
    except Exception as e:
        print(e)
    finally:
        final_ckpt_path = f'ckpt/{config.scene_name}/{get_current_function_name()}'
        saved_ckpt = model.save_ckpt(final_ckpt_path, final=False)
        writer.close()
        return saved_ckpt


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    import argparse

    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument('--option', type=str, choices=['train_density_only', 'train_velocity', 'train_velocity_lcc', 'train_joint', 'train_joint_lcc'], required=True, help="[Required][General] Choose the operation to execute.")
    parser.add_argument('--tag', type=str, default="DEFAULT", help="[General] Tag for the training run.")
    parser.add_argument('--device', type=str, default="cuda:0", help="[General] Device to run the operation.")
    parser.add_argument('--dtype', type=str, default="float32", choices=['float32', 'float16'], help="[General] Data type to use.")
    parser.add_argument('--scene', type=str, choices=['hyfluid', 'plume_1', 'plume_5', 'plume_10', 'plume_color_1'], default="hyfluid", help="[General] Scene to run.")
    parser.add_argument('--frame_start', type=int, default=0, help="[General] Start frame for training.")
    parser.add_argument('--frame_end', type=int, default=120, help="[General] End frame for training.")
    parser.add_argument('--batch_size', type=int, default=512, help="[General] Batch size for training.")
    parser.add_argument('--depth_size', type=int, default=512, help="[General] Depth size for training.")
    parser.add_argument('--ratio', type=float, default=0.5, help="[General] Ratio of resolution resampling.")
    parser.add_argument('--total_iter', type=int, default=10000, help="[General] Total iterations for training.")
    parser.add_argument('--select_ckpt', action='store_true', help="[General] Select a pretrained checkpoint file.")
    parser.add_argument('--checkpoint', type=str, default=None, help="[General] Load a pretrained checkpoint.")
    parser.add_argument('--mid_ckpt_iters', type=int, default=-1, help="[General] Save mid checkpoints during training.")
    parser.add_argument('--resx', type=int, default=128, help="[train_velocity only] Resolution in x direction.")
    parser.add_argument('--resy', type=int, default=192, help="[train_velocity only] Resolution in y direction.")
    parser.add_argument('--resz', type=int, default=128, help="[train_velocity only] Resolution in z direction.")
    parser.add_argument('--lw_img', type=float, default=1.0, help="[train_density_only, train_velocity only, train_joint] Weight for image loss.")
    parser.add_argument('--lw_nse', type=float, default=1.0, help="[train_velocity only, train_joint] Weight for NSE loss.")
    parser.add_argument('--lw_proj', type=float, default=1.0, help="[train_velocity only, train_joint] Weight for projection loss.")
    parser.add_argument('--lw_min_vel_reg', type=float, default=10.0, help="[train_velocity only, train_joint] Weight for minimum velocity regularization.")
    parser.add_argument('--lw_lcc', type=float, default=1.0, help="[train_velocity_lcc only, train_joint_lcc] Weight for LCC loss.")
    args = parser.parse_args()

    loss_dict = {
        "lw_img": args.lw_img,
        "lw_nse": args.lw_nse,
        "lw_proj": args.lw_proj,
        "lw_min_vel_reg": args.lw_min_vel_reg,
        "lw_lcc": args.lw_lcc,
    }

    if args.select_ckpt and args.checkpoint is None:
        args.checkpoint = open_file_dialog()

    args_str = ' '.join([f'--{k}={v}' for k, v in vars(args).items()])
    print(f"\033[32m==================== Running command: {args_str} ====================\033[0m")
    train_config = TrainConfig(
        train_script=args_str,
        train_tag=args.tag,
        scene_name=args.scene,
        target_device=torch.device(args.device),
        target_dtype=torch.float32 if args.dtype == "float32" else torch.float16,
        batch_size=args.batch_size,
        depth_size=args.depth_size,
        ratio=args.ratio,
        mid_ckpts_iters=args.mid_ckpt_iters,
        use_rgb=args.scene == "plume_color_1",
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        loss_dict=loss_dict,
    )

    print(f"\033[32m= Training On {train_config.scene_name} \033[0m")

    if args.option == "train_density_only":
        train_density_only(
            config=train_config,
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_velocity":
        train_velocity(
            config=train_config,
            pretrain_density=20000,
            resx=args.resx,
            resy=args.resy,
            resz=args.resz,
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_velocity_lcc":
        train_velocity_lcc(
            config=train_config,
            lcc_path=f'data/{args.scene}/{args.scene}_lcc.yaml',
            pretrain_density=20000,
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_joint":
        train_joint(
            config=train_config,
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    if args.option == "train_joint_lcc":
        train_joint_lcc(
            config=train_config,
            lcc_path=f'data/{args.scene}/{args.scene}_lcc.yaml',
            total_iter=args.total_iter,
            pretrained_ckpt=args.checkpoint,
        )

    print("\033[31m红色文字\033[0m")
    print("\033[32m绿色文字\033[0m")
    print("\033[33m黄色文字\033[0m")
    print("\033[34m蓝色文字\033[0m")
    print("\033[32m==================== Training completed. ====================\033[0m")
