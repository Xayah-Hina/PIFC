from train import *


def train_density_only(config: TrainConfig, total_iter, final=False, pretrained_ckpt=None):
    losses = []
    steps = []
    avg_losses = []
    avg_steps = []

    model = TrainDensityModel(config)
    if pretrained_ckpt:
        model.load_ckpt(pretrained_ckpt, config.target_device)
    import tqdm
    try:
        for _ in tqdm.trange(total_iter):
            img_loss = model.forward(config.batch_size, config.depth_size)
            losses.append(img_loss.item())
            steps.append(model.global_step)

            tqdm.tqdm.write(f"iter: {model.global_step}, lr_d: {model.scheduler_d.get_last_lr()[0]}, img_loss: {img_loss}")
            if len(steps) % 100 == 0:
                avg_loss = sum(losses[-100:]) / 100
                avg_losses.append(avg_loss)
                avg_steps.append(model.global_step)
    except Exception as e:
        print(e)
    finally:
        model.save_ckpt(f'ckpt/{config.scene_name}/train_density_only', final=final)

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%H")
        filename = f"loss_train_density_only_{timestamp}_bs{config.batch_size}_{model.global_step}.png"
        save_dir = f'ckpt/{config.scene_name}/image'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(avg_steps, avg_losses, label='Average Image Loss (every 100 steps)', color='blue', linestyle='-', marker='o')
        plt.xlabel('Global Step')
        plt.ylabel('Image Loss (Averaged)')
        plt.title('Average Image Loss vs. Global Step')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        plt.close()

        print(f"Image loss curve saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training or validation.")
    parser.add_argument('--option', type=str, choices=['train_density_only', 'train_velocity_only', 'train_joint', 'validate_sample_grid', 'validate_render_frame', 'resimulation'], required=True, help="Choose the operation to execute.")
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
            final=False,
            pretrained_ckpt=args.checkpoint
        )

    print("==================== Operation completed. ====================")
