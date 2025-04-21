import subprocess

HYTHON_EXE = "C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe"

def train_velocity_multiprocessing(scene, devices):
    tags = ["TAG1", "TAG2", "TAG3", "TAG4", "TAG5", "TAG6", "TAG7", "TAG8", "TAG9", "TAG10"]
    lw_imgs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_nses = [1.0, 5.0, 10.0, 100.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_projs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_min_vel_regs = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    lw_lccs = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
    option = "train_velocity"
    total_iter = 3000
    frame_start = 0
    frame_end = 120
    checkpoint = "history/train_density_only/ckpt_plume_1_cuda1_0419204709_100000.tar"

    processes = []
    device_num = len(devices)
    device_iter = 0
    for tag, lw_img, lw_nse, lw_proj, lw_min_vel_reg, lw_lcc in zip(tags, lw_imgs, lw_nses, lw_projs, lw_min_vel_regs, lw_lccs):
        print(f"Training {tag}")
        p = subprocess.Popen([HYTHON_EXE, "run_train.py",
                              f"--tag={tag}",
                              f"--option={option}",
                              f"--scene={scene}",
                              f"--total_iter={total_iter}",
                              f"--frame_start={frame_start}",
                              f"--frame_end={frame_end}",
                              f"--lw_img={lw_img}",
                              f"--lw_nse={lw_nse}",
                              f"--lw_proj={lw_proj}",
                              f"--lw_min_vel_reg={lw_min_vel_reg}",
                              f"--lw_lcc={lw_lcc}",
                              f"--checkpoint={checkpoint}",
                              f"--device={devices[device_iter % device_num]}"])
        device_iter += 1
        processes.append(p)
        import time
        time.sleep(2.0)

    for p in processes:
        p.wait()


def train_velocity_lcc_multiprocessing(scene, devices):
    tags = ["TAG1", "TAG2", "TAG3", "TAG4", "TAG5", "TAG6", "TAG7", "TAG8", "TAG9", "TAG10"]
    lw_imgs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_nses = [1.0, 5.0, 10.0, 100.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_projs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_min_vel_regs = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    lw_lccs = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
    option = "train_velocity_lcc"
    total_iter = 3000
    frame_start = 0
    frame_end = 120
    checkpoint = "history/train_density_only/ckpt_plume_1_cuda1_0419204709_100000.tar"

    processes = []
    device_num = len(devices)
    device_iter = 0
    for tag, lw_img, lw_nse, lw_proj, lw_min_vel_reg, lw_lcc in zip(tags, lw_imgs, lw_nses, lw_projs, lw_min_vel_regs, lw_lccs):
        print(f"Training {tag}")
        p = subprocess.Popen([HYTHON_EXE, "run_train.py",
                              f"--tag={tag}",
                              f"--option={option}",
                              f"--scene={scene}",
                              f"--total_iter={total_iter}",
                              f"--frame_start={frame_start}",
                              f"--frame_end={frame_end}",
                              f"--lw_img={lw_img}",
                              f"--lw_nse={lw_nse}",
                              f"--lw_proj={lw_proj}",
                              f"--lw_min_vel_reg={lw_min_vel_reg}",
                              f"--lw_lcc={lw_lcc}",
                              f"--checkpoint={checkpoint}",
                              f"--device={devices[device_iter % device_num]}"])
        device_iter += 1
        processes.append(p)
        import time
        time.sleep(2.0)

    for p in processes:
        p.wait()


def export_velocity_field_multiprocessing(pretrained_ckpt_path, devices):
    option = "export_velocity_field"

    from pathlib import Path
    ckpt_files = list(Path(pretrained_ckpt_path).glob("*.tar"))
    assert len(ckpt_files) > 0, f"Checkpoint files not found in {pretrained_ckpt_path}"

    processes = []
    device_num = len(devices)
    device_iter = 0

    for checkpoint in ckpt_files:
        p = subprocess.Popen([HYTHON_EXE, "run_evaluation.py",
                              f"--option={option}",
                              f"--checkpoint={checkpoint}",
                              f"--device={devices[device_iter % device_num]}"])
        device_iter += 1
        processes.append(p)
        import time
        time.sleep(2.0)

    for p in processes:
        p.wait()


def evaluate_resimulation_multiprocessing(pretrained_ckpt_path, devices):
    option = "evaluate_resimulation"

    from pathlib import Path
    ckpt_files = list(Path(pretrained_ckpt_path).glob("*.tar"))
    assert len(ckpt_files) > 0, f"Checkpoint files not found in {pretrained_ckpt_path}"

    processes = []
    device_num = len(devices)
    device_iter = 0

    for checkpoint in ckpt_files:
        p = subprocess.Popen([HYTHON_EXE, "run_evaluation.py",
                              f"--option={option}",
                              f"--checkpoint={checkpoint}",
                              f"--device={devices[device_iter % device_num]}"])
        device_iter += 1
        processes.append(p)
        import time
        time.sleep(2.0)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    DEVICES = ["cuda:0", "cuda:1"]
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple training scripts.")
    parser.add_argument('--option', type=str, choices=['train_velocity_multiprocessing', 'evaluate_velocity_multiprocessing', 'train_velocity_lcc_multiprocessing'], required=True, help="[Required][General] Choose the operation to execute.")
    parser.add_argument('--scene', type=str, choices=['hyfluid', 'plume_1', 'plume_color_1'], default="hyfluid", help="[General] Scene to run.")
    args = parser.parse_args()

    if args.option == "train_velocity_multiprocessing":
        train_velocity_multiprocessing(scene=args.scene, devices=DEVICES)
        export_velocity_field_multiprocessing(pretrained_ckpt_path=f"ckpt/{args.scene}/train_velocity", devices=DEVICES)
        evaluate_resimulation_multiprocessing(pretrained_ckpt_path=f"ckpt/{args.scene}/train_velocity", devices=DEVICES)

    if args.option == "train_velocity_lcc_multiprocessing":
        train_velocity_lcc_multiprocessing(scene=args.scene, devices=DEVICES)
        export_velocity_field_multiprocessing(pretrained_ckpt_path=f"ckpt/{args.scene}/train_velocity_lcc", devices=DEVICES)
        evaluate_resimulation_multiprocessing(pretrained_ckpt_path=f"ckpt/{args.scene}/train_velocity_lcc", devices=DEVICES)

    if args.option == "evaluate_velocity_multiprocessing":
        export_velocity_field_multiprocessing(pretrained_ckpt_path=f"ckpt/{args.scene}/train_velocity", devices=DEVICES)
