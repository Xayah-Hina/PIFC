import hou
import torch
import os


### Please use Hython instead of common Python
### You can find Hython under Houdini Installed dir
### Eg. "C:/Program Files/Side Effects Software/Houdini 20.5.550/bin/hython.exe"

def export_density_field(den, save_path, surname, local2world, scale):
    transform_matrix = hou.Matrix4([v for row in local2world.tolist() for v in row])
    scale_matrix = hou.hmath.buildScale(scale.tolist())
    final_matrix = transform_matrix * scale_matrix

    resx, resy, resz = den.shape[0], den.shape[1], den.shape[2]
    geo = hou.Geometry()
    vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    vol.setAllVoxels(den.cpu().numpy().flatten().tolist())
    vol.setTransform(final_matrix)
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")


def export_density_field_with_bbox(den, save_path, surname, local2world, scale, bbox_min, bbox_max):
    transform_matrix = hou.Matrix4([v for row in local2world.tolist() for v in row])
    scale_matrix = hou.hmath.buildScale(scale.tolist())
    final_matrix = transform_matrix * scale_matrix
    final_matrix = hou.Matrix4(1)

    resx, resy, resz = den.shape[0], den.shape[1], den.shape[2]
    geo = hou.Geometry()
    vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    vol.setAllVoxels(den.cpu().numpy().flatten().tolist())
    vol.setTransform(final_matrix)

    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")

    geo = hou.Geometry()

    # 归一化后变换到 world space
    def voxel_to_world(x, y, z):
        pos = hou.Vector3(float(x) / resx, float(y) / resy, float(z) / resz)
        return pos * final_matrix.transposed()

    # 解包 bbox 坐标
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    # 8 个点
    p000 = geo.createPoint()
    p000.setPosition(voxel_to_world(x0, y0, z0))  # min, min, min
    p100 = geo.createPoint()
    p100.setPosition(voxel_to_world(x1, y0, z0))  # max, min, min
    p010 = geo.createPoint()
    p010.setPosition(voxel_to_world(x0, y1, z0))  # min, max, min
    p110 = geo.createPoint()
    p110.setPosition(voxel_to_world(x1, y1, z0))  # max, max, min
    p001 = geo.createPoint()
    p001.setPosition(voxel_to_world(x0, y0, z1))  # min, min, max
    p101 = geo.createPoint()
    p101.setPosition(voxel_to_world(x1, y0, z1))  # max, min, max
    p011 = geo.createPoint()
    p011.setPosition(voxel_to_world(x0, y1, z1))  # min, max, max
    p111 = geo.createPoint()
    p111.setPosition(voxel_to_world(x1, y1, z1))  # max, max, max

    # bottom face (z = z0)
    poly_bottom = geo.createPolygon(False)
    poly_bottom.addVertex(p000)
    poly_bottom.addVertex(p100)
    poly_bottom.addVertex(p110)
    poly_bottom.addVertex(p010)

    # top face (z = z1)
    poly_top = geo.createPolygon(False)
    poly_top.addVertex(p001)
    poly_top.addVertex(p101)
    poly_top.addVertex(p111)
    poly_top.addVertex(p011)

    # front face (y = y0)
    poly_front = geo.createPolygon(False)
    poly_front.addVertex(p000)
    poly_front.addVertex(p001)
    poly_front.addVertex(p101)
    poly_front.addVertex(p100)

    # back face (y = y1)
    poly_back = geo.createPolygon(False)
    poly_back.addVertex(p010)
    poly_back.addVertex(p011)
    poly_back.addVertex(p111)
    poly_back.addVertex(p110)

    # left face (x = x0)
    poly_left = geo.createPolygon(False)
    poly_left.addVertex(p000)
    poly_left.addVertex(p010)
    poly_left.addVertex(p011)
    poly_left.addVertex(p001)

    # right face (x = x1)
    poly_right = geo.createPolygon(False)
    poly_right.addVertex(p100)
    poly_right.addVertex(p101)
    poly_right.addVertex(p111)
    poly_right.addVertex(p110)

    output_path = os.path.join(save_path, f"{surname}_bbox.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")


def export_velocity_field(vel, save_path, surname, local2world, scale):
    transform_matrix = hou.Matrix4([v for row in local2world.tolist() for v in row])
    scale_matrix = hou.hmath.buildScale(scale.tolist())
    final_matrix = transform_matrix * scale_matrix

    resx, resy, resz = vel.shape[0], vel.shape[1], vel.shape[2]
    geo = hou.Geometry()
    vel_np = vel.cpu().numpy()
    name_attrib = geo.addAttrib(hou.attribType.Prim, "name", "default")
    for i, name in enumerate(['vel.x', 'vel.y', 'vel.z']):
        vol = geo.createVolume(resx, resy, resz, hou.BoundingBox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
        vol.setAttribValue(name_attrib, name)
        data_flat = vel_np[..., i].flatten().tolist()
        vol.setAllVoxels(data_flat)
        vol.setTransform(final_matrix)
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, f"{surname}.bgeo.sc")
    geo.saveToFile(output_path)
    print(f"Save {output_path}")


def create_voxel_boxes(occupancy_grid, save_path, surname, local2world, scale):
    geo = hou.Geometry()
    scale_i = (1.0 / occupancy_grid.shape[0], 1.0 / occupancy_grid.shape[1], 1.0 / occupancy_grid.shape[2])
    transform_matrix = hou.Matrix4([v for row in local2world.tolist() for v in row])
    scale_matrix = hou.hmath.buildScale(scale.tolist())
    final_matrix = transform_matrix * scale_matrix

    print(f"Creating voxel boxes... (occupancy_grid shape: {occupancy_grid.shape})")
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            for k in range(occupancy_grid.shape[2]):
                if occupancy_grid[i, j, k]:
                    point_000 = hou.Vector3(i * scale_i[0], j * scale_i[1], k * scale_i[2]) * final_matrix.transposed()
                    point_001 = hou.Vector3((i + 1) * scale_i[0], j * scale_i[1], k * scale_i[2]) * final_matrix.transposed()
                    point_010 = hou.Vector3(i * scale_i[0], (j + 1) * scale_i[1], k * scale_i[2]) * final_matrix.transposed()
                    point_011 = hou.Vector3(i * scale_i[0], (j + 1) * scale_i[1], (k + 1) * scale_i[2]) * final_matrix.transposed()
                    point_100 = hou.Vector3((i + 1) * scale_i[0], j * scale_i[1], (k + 1) * scale_i[2]) * final_matrix.transposed()
                    point_101 = hou.Vector3((i + 1) * scale_i[0], (j + 1) * scale_i[1], (k + 1) * scale_i[2]) * final_matrix.transposed()
                    point_110 = hou.Vector3((i + 1) * scale_i[0], (j + 1) * scale_i[1], k * scale_i[2]) * final_matrix.transposed()
                    point_111 = hou.Vector3((i + 1) * scale_i[0], (j + 1) * scale_i[1], (k + 1) * scale_i[2]) * final_matrix.transposed()

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
