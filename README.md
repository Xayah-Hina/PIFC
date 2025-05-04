# Physics Informed Fluid Control

## Environment Setup
```shell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install tensorboard tqdm pyyaml av imageio imageio[ffmpeg] pykalman scipy phiflow
python -m pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp311-cp311-win_amd64.whl
```

## Checkpoints

### Scene HyFluid

- [ckpt_hyfluid_cuda0_0419205209_100000.tar](history/density/ckpt_hyfluid_cuda0_0419205209_100000.tar): BASIC density only
- [ckpt_hyfluid_cuda0_0417184733_110000.tar](history/joint/ckpt_hyfluid_cuda0_0417184733_110000.tar): BASIC joint

### Scene Plume 10

- [[DEFAULT]_ckpt_plume_10_cuda1_0504044531_0_120_010000.tar](history/density/%5BDEFAULT%5D_ckpt_plume_10_cuda1_0504044531_0_120_010000.tar): BASIC density only
- [[DEFAULT]_ckpt_plume_10_cuda0_0504043937_0_120_010000.tar](history/density/%5BDEFAULT%5D_ckpt_plume_10_cuda0_0504043937_0_120_010000.tar): BASIC density only (NOT TESTED)
- [[DEFAULT]_ckpt_plume_10_cuda1_0504063000_0_120_020000.tar](history/joint/%5BDEFAULT%5D_ckpt_plume_10_cuda1_0504063000_0_120_020000.tar): BASIC joint
- [[[10-0]]_ckpt_plume_10_cuda1_0504090440_0_120_020000.tar](history/joint/%5B%5B10-0%5D%5D_ckpt_plume_10_cuda1_0504090440_0_120_020000.tar): BASIC joint (validation)
- [[[10-3]]_ckpt_plume_10_cuda0_0504091203_0_120_020000.tar](history/joint/%5B%5B10-3%5D%5D_ckpt_plume_10_cuda0_0504091203_0_120_020000.tar): joint + zero reg