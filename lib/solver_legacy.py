import torch
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


def advect_SL(q_grid, vel_world_prev, coord_3d_sim, dt, RK=1):
    """Advect a scalar quantity using a given velocity field.
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
        RK: int, number of Runge-Kutta steps
        y_start: where to start at y-axis
        proj_y: simulation domain resolution at y-axis
        use_project: whether to use Poisson solver
        project_solver: Poisson solver
        bbox_model: bounding box model
    Returns:
        advected_quantity: [X, Y, Z, 1]
        vel_world: [X, Y, Z, 3]
    """
    if RK == 1:
        vel_world = vel_world_prev.clone()
        vel_sim = vel_world  # [X, Y, Z, 3]
    elif RK == 2:
        vel_world = vel_world_prev.clone()  # [X, Y, Z, 3]
        # breakpoint()
        vel_sim = vel_world  # [X, Y, Z, 3]
        coord_3d_sim_midpoint = coord_3d_sim - 0.5 * dt * vel_sim  # midpoint
        midpoint_sampled = coord_3d_sim_midpoint * 2 - 1  # [X, Y, Z, 3]
        vel_sim = torch.nn.functional.grid_sample(vel_sim.permute(3, 2, 1, 0)[None], midpoint_sampled.permute(2, 1, 0, 3)[None], align_corners=True, padding_mode='zeros').squeeze(0).permute(3, 2, 1, 0)  # [X, Y, Z, 3]
    else:
        raise NotImplementedError
    backtrace_coord = coord_3d_sim - dt * vel_sim  # [X, Y, Z, 3]
    backtrace_coord_sampled = backtrace_coord * 2 - 1  # ranging [-1, 1]
    q_grid = q_grid[None, ...].permute([0, 4, 3, 2, 1])  # [N, C, Z, Y, X] i.e., [N, C, D, H, W]
    q_backtraced = torch.nn.functional.grid_sample(q_grid, backtrace_coord_sampled.permute(2, 1, 0, 3)[None, ...], align_corners=True, padding_mode='zeros')  # [N, C, D, H, W]
    q_backtraced = q_backtraced.squeeze(0).permute([3, 2, 1, 0])  # [X, Y, Z, C]
    return q_backtraced


def advect_maccormack(q_grid, vel_sim_prev, coord_3d_sim, dt):
    """
    Args:
        q_grid: [X', Y', Z', C]
        vel_world_prev: [X, Y, Z, 3]
        coord_3d_sim: [X, Y, Z, 3]
        dt: float
    Returns:
        advected_quantity: [X, Y, Z, C]
        vel_world: [X, Y, Z, 3]
    """
    q_grid_next = advect_SL(q_grid, vel_sim_prev, coord_3d_sim, dt)
    q_grid_back = advect_SL(q_grid_next, vel_sim_prev, coord_3d_sim, -dt)
    q_advected = q_grid_next + (q_grid - q_grid_back) / 2
    C = q_advected.shape[-1]
    for i in range(C):
        q_max, q_min = q_grid[..., i].max(), q_grid[..., i].min()
        q_advected[..., i] = q_advected[..., i].clamp_(q_min, q_max)
    return q_advected


@jit_compile
def advect_step(v, s, dt):
    s = advect.mac_cormack(s, v, dt)
    return s


if __name__ == '__main__':
    torch.set_printoptions(precision=20)
    np.set_printoptions(precision=20)

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
    den = torch.from_numpy(np.load(file_den0)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)

    for step in tqdm.trange(120):
        mask_to_sim = coord_3d_sim[..., 1] > source_height
        if step > 0:
            file_vel = f'../houdini/fields/vel_hy/vel_{step-1:03d}.npy'
            vel = torch.from_numpy(np.load(file_vel)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)
            confinement_field = torch.zeros_like(vel)
            vel_confined = vel + confinement_field
            vel_sim_confined = BBOX_MODEL_gpu.world2sim_rot(vel_confined)
            den = advect_maccormack(den, vel_sim_confined, coord_3d_sim, dt)

            file_den_ori = f'../houdini/fields/den_hy/den_{step:03d}.npy'
            den_ori = torch.from_numpy(np.load(file_den_ori)).to(torch.device('cuda:0')).permute(2, 1, 0, 3)
            den[~mask_to_sim] = den_ori[~mask_to_sim]
            den[~bbox_mask] *= 0.
        os.makedirs('output_test', exist_ok=True)
        np.save(os.path.join('output_test', f'density_advected_{step:03d}.npy'), den.cpu().numpy())

    # domain = Box(x=0.4909, y=0.73635, z=0.4909)
    # res = (128, 192, 128)
    #
    # file_den0 = '../houdini/fields/den/density_000.npz'
    # den_native_tensor = torch.from_numpy(np.load(file_den0)['den']).to(torch.device('cuda:0')).squeeze(-1)
    # den_phi_tensor = math.tensor(den_native_tensor, spatial('x,y,z'))
    # source = CenteredGrid(den_phi_tensor, ZERO_GRADIENT, x=res[0], y=res[1], z=res[2], bounds=Box(x=108, y=192, z=108))
    # source.data.native('x,y,z')
    #
    # output_dir = '../houdini/fields/output'
    # os.makedirs(output_dir, exist_ok=True)
    #
    # smoke = source
    # dt = 1. / 120.
    # print(dt)
    # for step in tqdm.trange(120):
    #     file_vel0 = f'../houdini/fields/vel/velocity_{step:03d}.npz'
    #     vel_native_tensor = torch.from_numpy(np.load(file_vel0)['vel']).to(torch.device('cuda:0'))
    #     vel_phi_tensor = math.tensor(vel_native_tensor, spatial('x,y,z'), channel('vector'))
    #     velocity = StaggeredGrid(vel_phi_tensor, 0, x=res[0], y=res[1], z=res[2], bounds=Box(x=108, y=192, z=108))
    #     smoke = advect_step(velocity, smoke, dt)
    #     np.savez_compressed(os.path.join(output_dir, f'density_advected_{step:03d}.npz'), den=smoke.data.native('x,y,z').cpu().numpy())
    #     print(f'Complete Step {step:03d}')
