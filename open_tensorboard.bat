@echo off
set "CKPT=%~1"

REM 切换到当前 .bat 文件所在的目录
cd /d %~dp0

REM 进入上一级目录
cd ..

REM open tensorboard
tensorboard --logdir=ckpt

REM 可选：运行完暂停窗口
pause