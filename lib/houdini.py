import hou
import os


def export_density_field(den, save_path, surname):
    resx, resy, resz = den.shape[0], den.shape[1], den.shape[2]
    geo = hou.Geometry()
    vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(0.0, 0.0, 0.0, 0.4909, 0.73635, 0.4909))
    vol.setAllVoxels(den.cpu().numpy().flatten().tolist())
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")
