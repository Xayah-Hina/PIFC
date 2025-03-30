from phi.torch.flow import *
import numpy as np
import os
import tqdm


def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3, :3]), -1)  # 4.world to 3.target
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    pos_scale = new_pose / (scale_vector)  # 3.target to 2.simulation
    return pos_scale


class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=[0.15, 0.0, 0.15], in_max=[0.85, 1., 0.85], device=torch.device("cuda:0")):
        self.s_w2s = torch.tensor(smoke_tran_inv, device=device, dtype=torch.float32).expand([4, 4])
        self.s2w = torch.inverse(self.s_w2s)
        self.s_scale = torch.tensor(smoke_scale.copy(), device=device, dtype=torch.float32).expand([3])
        self.s_min = torch.tensor(in_min, device=device, dtype=torch.float32)
        self.s_max = torch.tensor(in_max, device=device, dtype=torch.float32)

    def world2sim(self, pts_world):
        pts_world_homo = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)
        pts_sim_ = torch.matmul(self.s_w2s, pts_world_homo[..., None]).squeeze(-1)[..., :3]
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def world2sim_rot(self, pts_world):
        pts_sim_ = torch.matmul(self.s_w2s[:3, :3], pts_world[..., None]).squeeze(-1)
        pts_sim = pts_sim_ / (self.s_scale)  # 3.target to 2.simulation
        return pts_sim

    def sim2world(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_sim_homo = torch.cat([pts_sim_, torch.ones_like(pts_sim_[..., :1])], dim=-1)
        pts_world = torch.matmul(self.s2w, pts_sim_homo[..., None]).squeeze(-1)[..., :3]
        return pts_world

    def sim2world_rot(self, pts_sim):
        pts_sim_ = pts_sim * self.s_scale
        pts_world = torch.matmul(self.s2w[:3, :3], pts_sim_[..., None]).squeeze(-1)
        return pts_world

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[..., 0] >= self.s_min[0], target_pts[..., 1] >= self.s_min[1])
        above = torch.logical_and(above, target_pts[..., 2] >= self.s_min[2])
        below = torch.logical_and(target_pts[..., 0] <= self.s_max[0], target_pts[..., 1] <= self.s_max[1])
        below = torch.logical_and(below, target_pts[..., 2] <= self.s_max[2])
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts, to_float=True):
        return self.isInside(inputs_pts).to(torch.float) if to_float else self.isInside(inputs_pts)


@jit_compile
def advect_step(v, s, dt):
    s = advect.mac_cormack(s, v, dt)
    return s


if __name__ == '__main__':
    # domain = Box(x=0.4909, y=0.73635, z=0.4909)
    domain = Box(x=1., y=1., z=1.)

    VOXEL_TRAN_np = np.array([[1.0000000e+00, 0.0000000e+00, 7.5497901e-08, 8.1816666e-02], [0.0000000e+00, 1.0000000e+00, 0.0000000e+00, -4.4627272e-02], [7.5497901e-08, 0.0000000e+00, -1.0000000e+00, -4.9089999e-03], [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    VOXEL_SCALE_np = np.array([0.4909, 0.73635, 0.4909])
    voxel_tran_inv = np.linalg.inv(VOXEL_TRAN_np)
    BBOX_MODEL_gpu = BBox_Tool(voxel_tran_inv, VOXEL_SCALE_np)

    rx, ry, rz = 128, 192, 128
    xs, ys, zs = torch.meshgrid([torch.linspace(0, 1, rx), torch.linspace(0, 1, ry), torch.linspace(0, 1, rz)], indexing='ij')
    coord_3d_sim = torch.stack([xs, ys, zs], dim=-1).to(torch.device('cuda:0'))
    coord_3d_world = BBOX_MODEL_gpu.sim2world(coord_3d_sim)

    bbox_mask = BBOX_MODEL_gpu.insideMask(coord_3d_world[..., :3].reshape(-1, 3), to_float=False)
    bbox_mask = bbox_mask.reshape(rx, ry, rz)

    dt = 1. / 119.

    source_height = 0.25

    file_den0 = '../houdini/fields/den_hy/den_000.npy'
    den_native_tensor = torch.from_numpy(np.load(file_den0)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)
    res = (128, 192, 128)
    den_phi_tensor = math.tensor(den_native_tensor.squeeze(-1), spatial('x,y,z'))
    smoke = CenteredGrid(den_phi_tensor, ZERO_GRADIENT, x=res[0], y=res[1], z=res[2], bounds=domain)

    for step in tqdm.trange(120):
        mask_to_sim = coord_3d_sim[..., 1] > source_height

        if step > 0:
            file_vel = f'../houdini/fields/vel_hy/vel_{step-1:03d}.npy'
            vel = torch.from_numpy(np.load(file_vel)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)
            confinement_field = torch.zeros_like(vel)
            vel_confined = vel + confinement_field
            vel_sim_confined = BBOX_MODEL_gpu.world2sim_rot(vel_confined)

            vel_phi_tensor = math.tensor(vel_sim_confined, spatial('x,y,z'), channel('vector'))
            velocity = StaggeredGrid(vel_phi_tensor, 0, x=res[0], y=res[1], z=res[2], bounds=domain)
            smoke = advect_step(velocity, smoke, dt)
            den_native_tensor = smoke.data.native('x,y,z').unsqueeze(-1)

            file_den_ori = f'../houdini/fields/den_hy/den_{step:03d}.npy'
            den_ori = torch.from_numpy(np.load(file_den_ori)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)
            den_native_tensor[~mask_to_sim] = den_ori[~mask_to_sim]
            den_native_tensor[~bbox_mask] *= 0.
            den_phi_tensor = math.tensor(den_native_tensor.squeeze(-1), spatial('x,y,z'))
            smoke = CenteredGrid(den_phi_tensor, ZERO_GRADIENT, x=res[0], y=res[1], z=res[2], bounds=domain)

        os.makedirs('output_test_phi', exist_ok=True)
        den_native_tensor = smoke.data.native('x,y,z').unsqueeze(-1)
        np.save(os.path.join('output_test_phi', f'density_advected_{step:03d}.npy'), den_native_tensor.cpu().numpy())
