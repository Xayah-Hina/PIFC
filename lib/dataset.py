import torch
import torchvision.io as io
import os
from pathlib import Path

# HyFluid Scene
training_videos_hyfluid = [
    "data/hyfluid/train00.mp4",
    "data/hyfluid/train01.mp4",
    "data/hyfluid/train02.mp4",
    "data/hyfluid/train03.mp4",
    "data/hyfluid/train04.mp4",
]
scene_min_hyfluid = [-0.132113, -0.103114, -0.753138]
scene_max_hyfluid = [0.773877, 0.99804, 0.186818]

camera_calibrations_hyfluid = [
    "data/hyfluid/cam_train00.npz",
    "data/hyfluid/cam_train01.npz",
    "data/hyfluid/cam_train02.npz",
    "data/hyfluid/cam_train03.npz",
    "data/hyfluid/cam_train04.npz",
]

training_videos = training_videos_hyfluid
camera_calibrations = camera_calibrations_hyfluid
scene_min_current = scene_min_hyfluid
scene_max_current = scene_max_hyfluid


def find_relative_paths(relative_path_list):
    current_dir = Path.cwd()
    search_dirs = [current_dir, current_dir.parent, current_dir.parent.parent]

    for i in range(len(relative_path_list)):
        found = False
        relative_path = relative_path_list[i]
        for directory in search_dirs:
            full_path = directory / relative_path
            if full_path.exists():
                relative_path_list[i] = str(full_path.resolve())
                found = True
                break

        if not found:
            raise FileNotFoundError(f"file not found: {relative_path}")


find_relative_paths(training_videos)
find_relative_paths(camera_calibrations)


def load_videos_data(*video_paths, ratio: float, dtype: torch.dtype):
    """
    Load multiple videos directly from given paths onto the specified device, resample images by ratio.

    Args:
    - *paths: str (arbitrary number of video file paths)

    Returns:
    - torch.Tensor of shape (T, V, H * ratio, W * ratio, C)
    """

    if not video_paths:
        raise ValueError("No video paths provided.")

    valid_paths = []
    for video_path in video_paths:
        _path = os.path.normpath(video_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Video path {_path} does not exist.")
        valid_paths.append(_path)

    _frames_tensors = []
    for _path in valid_paths:
        try:
            _frames, _, _ = io.read_video(_path, pts_unit="sec")
            _frames = _frames.to(dtype=dtype) / 255.0
            _frames_tensors.append(_frames)
        except Exception as e:
            print(f"Error loading video '{_path}': {e}")

    videos = torch.stack(_frames_tensors)

    V, T, H, W, C = videos.shape
    videos_permuted = videos.permute(0, 1, 4, 2, 3).reshape(V * T, C, H, W)
    new_H, new_W = int(H * ratio), int(W * ratio)
    videos_resampled = torch.nn.functional.interpolate(videos_permuted, size=(new_H, new_W), mode='bilinear', align_corners=False)
    videos_resampled = videos_resampled.reshape(V, T, C, new_H, new_W).permute(1, 0, 3, 4, 2)

    return videos_resampled


def load_cameras_data(*cameras_paths, ratio, device: torch.device, dtype: torch.dtype):
    """
    Load multiple camera calibration files directly from given paths onto the specified device.

    Args:
    - *cameras_paths: str (arbitrary number of camera calibration file paths)

    Returns:
    - poses: torch.Tensor of shape (N, 4, 4)
    - focals: torch.Tensor of shape (N)
    - width: torch.Tensor of shape (N)
    - height: torch.Tensor of shape (N)
    - near: torch.Tensor of shape (N)
    - far: torch.Tensor of shape (N)
    """

    if not cameras_paths:
        raise ValueError("No cameras paths provided.")

    valid_paths = []
    for camera_path in cameras_paths:
        _path = os.path.normpath(camera_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Camera path {_path} does not exist.")
        valid_paths.append(_path)

    import numpy as np
    camera_infos = [np.load(path) for path in valid_paths]
    widths = [int(info["width"]) for info in camera_infos]
    assert len(set(widths)) == 1, f"Error: Inconsistent widths found: {widths}. All cameras must have the same resolution."
    heights = [int(info["height"]) for info in camera_infos]
    assert len(set(heights)) == 1, f"Error: Inconsistent heights found: {heights}. All cameras must have the same resolution."
    nears = [float(info["near"]) for info in camera_infos]
    assert len(set(nears)) == 1, f"Error: Inconsistent nears found: {nears}. All cameras must have the same near plane."
    fars = [float(info["far"]) for info in camera_infos]
    assert len(set(fars)) == 1, f"Error: Inconsistent fars found: {fars}. All cameras must have the same far plane."
    poses = torch.stack([torch.tensor(info["cam_transform"], device=device, dtype=dtype) for info in camera_infos])
    focals = torch.tensor([info["focal"] * widths[0] / info["aperture"] for info in camera_infos], device=device, dtype=dtype)
    widths = torch.tensor(widths, device=device, dtype=torch.int32)
    heights = torch.tensor(heights, device=device, dtype=torch.int32)
    nears = torch.tensor(nears, device=device, dtype=dtype)
    fars = torch.tensor(fars, device=device, dtype=dtype)

    focals = focals * ratio
    widths = widths * ratio
    heights = heights * ratio

    return poses, focals, widths, heights, nears, fars


def load_rotating_camera_data(*cameras_paths, ratio, device: torch.device, dtype: torch.dtype):
    if not cameras_paths:
        raise ValueError("No cameras paths provided.")

    valid_paths = []
    for camera_path in cameras_paths:
        _path = os.path.normpath(camera_path)
        if not Path(_path).exists():
            raise FileNotFoundError(f"Camera path {_path} does not exist.")
        valid_paths.append(_path)

    import numpy as np
    camera_infos = [np.load(path) for path in valid_paths]
    poses = torch.stack([torch.tensor(info["cam_transform"], device=device, dtype=dtype) for info in camera_infos])
    focals = torch.stack([torch.tensor(info["focal"] * info["width"] / info["aperture"], device=device, dtype=dtype) for info in camera_infos])
    widths = torch.stack([torch.tensor(info["width"], device=device, dtype=torch.int32) for info in camera_infos])
    heights = torch.stack([torch.tensor(info["height"], device=device, dtype=torch.int32) for info in camera_infos])
    nears = torch.stack([torch.tensor(info["near"], device=device, dtype=dtype) for info in camera_infos])
    fars = torch.stack([torch.tensor(info["far"], device=device, dtype=dtype) for info in camera_infos])

    focals = focals * ratio
    widths = widths * ratio
    heights = heights * ratio

    return poses, focals, widths, heights, nears, fars
