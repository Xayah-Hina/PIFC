import hou
import os


### Please use Hython instead of common Python
### You can find Hython under Houdini Installed dir
### Eg. "C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe"

def export_density_field(den, save_path, surname, bbox):
    resx, resy, resz = den.shape[0], den.shape[1], den.shape[2]
    geo = hou.Geometry()
    vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(*bbox))
    vol.setAllVoxels(den.cpu().numpy().flatten().tolist())
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")


def export_velocity_field(vel, save_path, surname, bbox):
    resx, resy, resz = vel.shape[0], vel.shape[1], vel.shape[2]
    geo = hou.Geometry()
    vel_np = vel.cpu().numpy()
    name_attrib = geo.addAttrib(hou.attribType.Prim, "name", "default")
    for i, name in enumerate(['vel.x', 'vel.y', 'vel.z']):
        vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(*bbox))
        vol.setAttribValue(name_attrib, name)
        data_flat = vel_np[..., i].flatten().tolist()
        vol.setAllVoxels(data_flat)
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")


def create_voxel_boxes(occupancy_grid, save_path, surname):
    geo = hou.Geometry()
    scale = (1.0 / occupancy_grid.shape[0], 1.0 / occupancy_grid.shape[1], 1.0 / occupancy_grid.shape[2])

    print(f"Create voxel boxes... occupancy_grid shape: {occupancy_grid.shape}")
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            for k in range(occupancy_grid.shape[2]):
                if occupancy_grid[i, j, k]:
                    point_000 = hou.Vector3(i * scale[0], j * scale[1], k * scale[2])
                    point_001 = hou.Vector3((i + 1) * scale[0], j * scale[1], k * scale[2])
                    point_010 = hou.Vector3(i * scale[0], (j + 1) * scale[1], k * scale[2])
                    point_011 = hou.Vector3((i + 1) * scale[0], (j + 1) * scale[1], k * scale[2])
                    point_100 = hou.Vector3(i * scale[0], j * scale[1], (k + 1) * scale[2])
                    point_101 = hou.Vector3((i + 1) * scale[0], j * scale[1], (k + 1) * scale[2])
                    point_110 = hou.Vector3(i * scale[0], (j + 1) * scale[1], (k + 1) * scale[2])
                    point_111 = hou.Vector3((i + 1) * scale[0], (j + 1) * scale[1], (k + 1) * scale[2])

                    point_000_hou = geo.createPoint()
                    point_000_hou.setPosition(point_000)
                    point_001_hou = geo.createPoint()
                    point_001_hou.setPosition(point_001)
                    point_010_hou = geo.createPoint()
                    point_010_hou.setPosition(point_010)
                    point_011_hou = geo.createPoint()
                    point_011_hou.setPosition(point_011)
                    point_100_hou = geo.createPoint()
                    point_100_hou.setPosition(point_100)
                    point_101_hou = geo.createPoint()
                    point_101_hou.setPosition(point_101)
                    point_110_hou = geo.createPoint()
                    point_110_hou.setPosition(point_110)
                    point_111_hou = geo.createPoint()
                    point_111_hou.setPosition(point_111)

                    poly_bottom = geo.createPolygon(False)
                    poly_bottom.addVertex(point_000_hou)
                    poly_bottom.addVertex(point_001_hou)
                    poly_bottom.addVertex(point_011_hou)
                    poly_bottom.addVertex(point_010_hou)

                    poly_top = geo.createPolygon(False)
                    poly_top.addVertex(point_100_hou)
                    poly_top.addVertex(point_101_hou)
                    poly_top.addVertex(point_111_hou)
                    poly_top.addVertex(point_110_hou)

                    poly_left = geo.createPolygon(False)
                    poly_left.addVertex(point_000_hou)
                    poly_left.addVertex(point_001_hou)
                    poly_left.addVertex(point_101_hou)
                    poly_left.addVertex(point_100_hou)

                    poly_right = geo.createPolygon(False)
                    poly_right.addVertex(point_010_hou)
                    poly_right.addVertex(point_011_hou)
                    poly_right.addVertex(point_111_hou)
                    poly_right.addVertex(point_110_hou)

                    poly_front = geo.createPolygon(False)
                    poly_front.addVertex(point_000_hou)
                    poly_front.addVertex(point_010_hou)
                    poly_front.addVertex(point_110_hou)
                    poly_front.addVertex(point_100_hou)

                    poly_back = geo.createPolygon(False)
                    poly_back.addVertex(point_001_hou)
                    poly_back.addVertex(point_011_hou)
                    poly_back.addVertex(point_111_hou)
                    poly_back.addVertex(point_101_hou)

    # Save the geometry to a file
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")
