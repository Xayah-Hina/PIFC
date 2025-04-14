# from train import *
from validate import *
from tkinter import filedialog, Tk
import torch
import sdl2
import os


def open_file_dialog():
    while True:
        root = Tk()
        root.withdraw()  # 隐藏主窗口
        file_path = filedialog.askopenfilename(title="Select a checkpoint file", initialdir="ckpt", filetypes=[("Checkpoint files", "*.tar")])
        root.destroy()

        if file_path and os.path.isfile(file_path):
            print("Successfully loaded checkpoint:", file_path)
            return file_path
        else:
            print("invalid file path, please select a valid checkpoint file.")


if __name__ == "__main__":
    scene = 'plume_1'
    target_device = torch.device("cuda:0")
    target_dtype = torch.float32

    pose = torch.tensor([[-6.5174e-01, 7.3241e-02, 7.5490e-01, 3.5361e+00],
                         [-6.9389e-18, 9.9533e-01, -9.6567e-02, 1.9000e+00],
                         [-7.5844e-01, -6.2937e-02, -6.4869e-01, -2.6511e+00],
                         [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]], device=torch.device(target_device), dtype=target_dtype)
    focal = torch.tensor(1303.6753, device=torch.device(target_device), dtype=target_dtype)
    width = 1080
    height = 1920
    depth_size = 192
    near = 2.5
    far = 5.4
    frame = 110
    ratio = 0.5

    resolution = (int(width * ratio), int(height * ratio))

    #####################################################################################
    model = ValidationModel(scene, target_device, target_dtype)
    model.load_ckpt(open_file_dialog(), target_device)
    rgb_map_final = model.render_frame(pose, focal, width, height, depth_size, near, far, frame, ratio)
    rgb8 = (255 * np.clip(rgb_map_final.cpu().numpy(), 0, 1)).astype(np.uint8)
    pixel_data = rgb8.tobytes()
    #####################################################################################

    sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO)
    window = sdl2.SDL_CreateWindow(
        b"PySDL2 RGB Display",
        sdl2.SDL_WINDOWPOS_CENTERED,
        sdl2.SDL_WINDOWPOS_CENTERED,
        resolution[0], resolution[1],
        sdl2.SDL_WINDOW_SHOWN | sdl2.SDL_WINDOW_RESIZABLE
    )
    renderer = sdl2.SDL_CreateRenderer(window, -1, sdl2.SDL_RENDERER_ACCELERATED)
    texture = sdl2.SDL_CreateTexture(
        renderer,
        sdl2.SDL_PIXELFORMAT_RGB24,  # 使用RGB格式
        sdl2.SDL_TEXTUREACCESS_STREAMING,
        resolution[0], resolution[1]
    )
    event = sdl2.SDL_Event()
    running = True

    while running:
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT:
                running = False
        sdl2.SDL_RenderClear(renderer)
        sdl2.SDL_UpdateTexture(texture, None, pixel_data, resolution[0] * 3)
        sdl2.SDL_RenderCopy(renderer, texture, None, None)
        sdl2.SDL_RenderPresent(renderer)

    sdl2.SDL_DestroyTexture(texture)
    sdl2.SDL_DestroyRenderer(renderer)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()
