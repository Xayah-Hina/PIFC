import subprocess


def train_velocity_multiprocessing(scene, devices):
    tags = ["TAG 1", "TAG 2", "TAG 3", "TAG 4", "TAG 5", "TAG 6", "TAG 7", "TAG 8", "TAG 9", "TAG 10"]
    lw_imgs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_nses = [1.0, 5.0, 10.0, 100.0, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_projs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lw_min_vel_regs = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    lw_lccs = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
    option = "train_velocity"
    total_iter = 2000
    frame_start = 0
    frame_end = 120
    checkpoint = "history/train_velocity/ckpt_hyfluid_cuda1_0420044255_150000.tar"

    processes = []

    device_num = len(devices)
    device_iter = 0
    for tag, lw_img, lw_nse, lw_proj, lw_min_vel_reg, lw_lcc in zip(tags, lw_imgs, lw_nses, lw_projs, lw_min_vel_regs, lw_lccs):
        print(f"Training {tag}")
        p = subprocess.Popen(["C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe", "run_train.py",
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
                              f"--device={devices[(device_iter + 1) % device_num]}"],
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple training scripts.")
    parser.add_argument('--option', type=str, choices=['train_velocity_multiprocessing'], required=True, help="[Required][General] Choose the operation to execute.")
    args = parser.parse_args()

    if args.option == "train_velocity_multiprocessing":
        train_velocity_multiprocessing(scene="plume_1", devices=["cuda:0", "cuda:1"])
