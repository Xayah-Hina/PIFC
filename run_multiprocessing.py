import subprocess


def train_velocity_multiprocessing(device):
    options = [
        "train_velocity",
        "train_velocity",
        "train_velocity",
        "train_velocity",
        "train_velocity",
    ]
    train_logs = [
        "[1]: train log",
        "[2]: train log",
        "[3]: train log",
        "[4]: train log",
        "[5]: train log",
    ]
    scenes = [
        "plume_1",
        "plume_1",
        "plume_1",
        "plume_1",
        "plume_1",
    ]
    total_iters = [
        2000,
        2000,
        2000,
        2000,
        2000,
    ]
    frame_starts = [
        0,
        0,
        0,
        0,
        0,
    ]
    frame_ends = [
        120,
        120,
        120,
        120,
        120,
    ]
    checkpoint = "history/train_velocity/ckpt_hyfluid_cuda1_0420044255_150000.tar"

    processes = []

    for opt, train_log, scene, total_iter, frame_start, frame_end in zip(options, train_logs, scenes, total_iters, frame_starts, frame_ends):
        p = subprocess.Popen(["C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe", "run_train.py", f"--option={opt}", f"--train_log={train_log}", f"--scene={scene}", f"--total_iter={total_iter}", f"--frame_start={frame_start}", f"--frame_end={frame_end}", f"--checkpoint={checkpoint}", f"--device={device}"])
        processes.append(p)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple training scripts.")
    parser.add_argument('--option', type=str, choices=['train_velocity_multiprocessing'], required=True, help="[Required][General] Choose the operation to execute.")
    parser.add_argument('--device', type=str, default="cuda:0", help="[General] Device to run the operation.")
    args = parser.parse_args()

    if args.option == "train_velocity_multiprocessing":
        train_velocity_multiprocessing(device=args.device)
