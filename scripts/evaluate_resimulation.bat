@echo off
set "CKPT=%~1"

REM 切换到当前 .bat 文件所在的目录
cd /d %~dp0

REM 进入上一级目录
cd ..

REM 固定参数 --option
set "OPTION=evaluate_resimulation"

REM 调用 Python 脚本
"C:/Program Files/Side Effects Software/Houdini 20.5.550/python311/python.exe" run_hyfluid.py --option=%OPTION% --checkpoint="%CKPT%"
"C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe" houdini/visualize_npz.py

REM 可选：运行完暂停窗口
pause