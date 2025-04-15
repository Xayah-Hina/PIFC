# Physics Informed Fluid Control

## Environment Setup
```shell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install tensorboard tqdm pyyaml av
python -m pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp311-cp311-win_amd64.whl
```

## Checkpoints

### Scene HyFluid

- `ckpt_040117_bs1024_200996.tar`: 100000 density only + 100000 velocity only
- `ckpt_040118_bs1024_100000.tar`: 100000 joint

### Scene Plume 1

- `ckpt_040300_bs1024_100000.tar`: 100000 density only