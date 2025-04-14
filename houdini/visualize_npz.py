import hou
import os
import glob
import numpy as np

input_dir = "ckpt/hyfluid/resimulation"
output_dir = "ckpt/hyfluid/resimulation"
os.makedirs(output_dir, exist_ok=True)

npy_files = sorted(glob.glob(os.path.join(input_dir, "density_advected_*.npy")))

print(f"Find {len(npy_files)} .npy files")

for npy_path in npy_files:
    filename = os.path.basename(npy_path)
    frame_str = filename.split("_")[-1].replace(".npy", "")

    data = np.load(npy_path).astype(np.float32)
    resx, resy, resz = data.shape[0], data.shape[1], data.shape[2]

    geo = hou.Geometry()
    vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(0.0, 0.0, 0.0, 0.4909, 0.73635, 0.4909))

    flat = data.flatten().tolist()
    vol.setAllVoxels(flat)

    output_path = os.path.join(output_dir, f"density_advected_{frame_str}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")
