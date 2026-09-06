import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# configuration
WIDTH, HEIGHT = 300, 300
NUM_AGENTS = 255
STEPS = 1000000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpsDipole:
    def __init__(self, w: int, h: int, device: torch.device):
        self.w = w
        self.h = h
        self.device = device

        # Layer 0: Upwash (Benefit / Lift) Field [1, 1, H, W]
        self.E_benefit_raw = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        self.E_benefit_act = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)

        # Layer 1: Downwash (Cost / Drag) Field [1, 1, H, W]
        self.E_cost_raw = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        self.E_cost_act = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)

        # Physics Parameters
        self.D = 0.28        # Diffusion rate
        self.decay = 0.88    # Short-lived wake dissipation
        self.dt = 1.0

        # Laplacian Stencil
        self.laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 3, 3)

    def step_physics(self, inj_benefit: torch.Tensor, inj_cost: torch.Tensor):
        """Diffuses and decays both benefit and cost fields with zero-gradient boundary padding."""
        # Benefit Field Physics
        padded_b = F.pad(self.E_benefit_act, (1, 1, 1, 1), mode="replicate")
        lap_b = F.conv2d(padded_b, self.laplacian_kernel, padding=0)
        self.E_benefit_raw = (self.E_benefit_act + self.D * lap_b * self.dt) * self.decay + inj_benefit
        self.E_benefit_act = torch.clamp(self.E_benefit_raw, 0.0, 1.0)

        # Cost Field Physics
        padded_c = F.pad(self.E_cost_act, (1, 1, 1, 1), mode="replicate")
        lap_c = F.conv2d(padded_c, self.laplacian_kernel, padding=0)
        self.E_cost_raw = (self.E_cost_act + self.D * lap_c * self.dt) * self.decay + inj_cost
        self.E_cost_act = torch.clamp(self.E_cost_raw, 0.0, 1.0)

    def sample_fields_and_gradients(self, pos: torch.Tensor):
        """Bilinear read of benefit and cost values along with benefit gradients for steering."""
        norm_x = (pos[:, 0] / (self.w - 1)) * 2.0 - 1.0
        norm_y = (pos[:, 1] / (self.h - 1)) * 2.0 - 1.0
        grid_coords = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)

        # Bilinear Sample Values
        val_b = F.grid_sample(self.E_benefit_raw, grid_coords, align_corners=True, mode="bilinear").view(-1)
        val_c = F.grid_sample(self.E_cost_raw, grid_coords, align_corners=True, mode="bilinear").view(-1)

        # Gradients of Benefit Field
        grad_bx = (torch.roll(self.E_benefit_raw, shifts=-1, dims=3) - torch.roll(self.E_benefit_raw, shifts=1, dims=3)) / 2.0
        grad_by = (torch.roll(self.E_benefit_raw, shifts=-1, dims=2) - torch.roll(self.E_benefit_raw, shifts=1, dims=2)) / 2.0

        gbx = F.grid_sample(grad_bx, grid_coords, align_corners=True, mode="bilinear").view(-1)
        gby = F.grid_sample(grad_by, grid_coords, align_corners=True, mode="bilinear").view(-1)

        # Gradients of Cost Field (For Downwash Repulsion)
        grad_cx = (torch.roll(self.E_cost_raw, shifts=-1, dims=3) - torch.roll(self.E_cost_raw, shifts=1, dims=3)) / 2.0
        grad_cy = (torch.roll(self.E_cost_raw, shifts=-1, dims=2) - torch.roll(self.E_cost_raw, shifts=1, dims=2)) / 2.0

        gcx = F.grid_sample(grad_cx, grid_coords, align_corners=True, mode="bilinear").view(-1)
        gcy = F.grid_sample(grad_cy, grid_coords, align_corners=True, mode="bilinear").view(-1)

        return val_b, val_c, torch.stack([gbx, gby], dim=-1), torch.stack([gcx, gcy], dim=-1)


class AerodynamicSwarm:
    def __init__(self, num_agents: int, w: int, h: int, device: torch.device):
        self.num_agents = num_agents
        self.w = w
        self.h = h
        self.device = device

        # Initialize agents clustered around center with aligned general heading
        self.pos = torch.rand((num_agents, 2), device=device) * (w * 0.4) + (w * 0.3)
        self.angles = torch.zeros(num_agents, device=device) + (np.pi / 4.0) + (torch.randn(num_agents, device=device) * 0.2)

        self.base_speed = 1.6
        self.fatigue = torch.rand(num_agents, device=device) * 0.3  # Internal energy budget state

    def update_and_inject(self, sps: SpsDipole):
        # 1. Read Perception (Temporal Separation)
        val_b, val_c, grad_b, grad_c = sps.sample_fields_and_gradients(self.pos)

        # 2. Update Fatigue State (Front pays full cost, flank riders save energy)
        # Base fatigue accumulation minus benefit read, plus cost read
        self.fatigue = torch.clamp(self.fatigue + 0.008 + (val_c * 0.02) - (val_b * 0.03), 0.0, 1.0)

        # 3. Dynamic Speed Modulation (Tired leaders slow down, letting rested flankers pass)
        effective_speed = self.base_speed * (1.0 - 0.45 * self.fatigue)

        # 4. Dipole Steering Policy: Attracted to Upwash (grad_b), Repelled by Downwash (grad_c)
        net_grad = grad_b * 2.0 - grad_c * 3.0
        grad_mag = torch.norm(net_grad, dim=-1)
        target_angles = torch.atan2(net_grad[:, 1], net_grad[:, 0])

        has_signal = grad_mag > 1e-3
        rand_turn = (torch.rand(self.num_agents, device=self.device) - 0.5) * 0.05

        self.angles = torch.where(
            has_signal,
            self.angles + 0.25 * torch.atan2(torch.sin(target_angles - self.angles), torch.cos(target_angles - self.angles)),
            self.angles + rand_turn,
        )

        # 5. Kinematic Step (with Periodic Boundary Wrap)
        dx = torch.cos(self.angles) * effective_speed
        dy = torch.sin(self.angles) * effective_speed
        self.pos[:, 0] = torch.remainder(self.pos[:, 0] + dx, self.w - 1.001)
        self.pos[:, 1] = torch.remainder(self.pos[:, 1] + dy, self.h - 1.001)

        # 6. Asymmetric Dipole Injection Splatting
        u = torch.stack([torch.cos(self.angles), torch.sin(self.angles)], dim=-1)
        p = torch.stack([-torch.sin(self.angles), torch.cos(self.angles)], dim=-1)

        # Dipole Offsets: Downwash (Behind), Upwash (Diagonal Flanks)
        p_downwash = self.pos - 4.0 * u
        p_left_upwash = self.pos - 4.0 * u - 3.5 * p
        p_right_upwash = self.pos - 4.0 * u + 3.5 * p

        inj_benefit = self._splat_positions(p_left_upwash, 0.4) + self._splat_positions(p_right_upwash, 0.4)
        inj_cost = self._splat_positions(p_downwash, 0.7)

        return inj_benefit, inj_cost

    def _splat_positions(self, positions: torch.Tensor, weight: float) -> torch.Tensor:
        """Bilinear splatting helper for a batch of continuous positions."""
        pos_clamped = torch.clamp(positions, 0, self.w - 1.001)
        ix = torch.floor(pos_clamped[:, 0]).long()
        iy = torch.floor(pos_clamped[:, 1]).long()
        dx_val = pos_clamped[:, 0] - ix.float()
        dy_val = pos_clamped[:, 1] - iy.float()

        inj_grid = torch.zeros((self.h, self.w), device=self.device)
        w_len = self.w

        w00 = (1.0 - dx_val) * (1.0 - dy_val) * weight
        w10 = dx_val * (1.0 - dy_val) * weight
        w01 = (1.0 - dx_val) * dy_val * weight
        w11 = dx_val * dy_val * weight

        inj_grid.put_(iy * w_len + ix, w00, accumulate=True)
        inj_grid.put_(iy * w_len + (ix + 1), w10, accumulate=True)
        inj_grid.put_((iy + 1) * w_len + ix, w01, accumulate=True)
        inj_grid.put_((iy + 1) * w_len + (ix + 1), w11, accumulate=True)

        return inj_grid.view(1, 1, self.h, self.w)


def run_dipole_demo():
    print(f"Executing Aerodynamic Dipole SPS Engine on: {DEVICE}")
    sps = SpsDipole(WIDTH, HEIGHT, DEVICE)
    swarm = AerodynamicSwarm(NUM_AGENTS, WIDTH, HEIGHT, DEVICE)

    cv2.namedWindow("SPS Dipole V-Formation Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SPS Dipole V-Formation Engine", 800, 800)

    start_time = time.time()
    for frame in range(STEPS):
        inj_b, inj_c = swarm.update_and_inject(sps)
        sps.step_physics(inj_b, inj_c)

        # Composite Visualization: Benefit (Green Channel) vs Cost (Red Channel)
        b_np = sps.E_benefit_act.squeeze().cpu().numpy()
        c_np = sps.E_cost_act.squeeze().cpu().numpy()

        vis_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        vis_frame[:, :, 1] = (b_np * 255).astype(np.uint8)  # Green = Upwash Lift Lobe
        vis_frame[:, :, 2] = (c_np * 255).astype(np.uint8)  # Red = Downwash Drag Lobe

        # Render Agents (Colored by Fatigue level: White = Fresh, Yellow/Red = Tired Lead)
        pos_np = swarm.pos.cpu().numpy()
        fatigue_np = swarm.fatigue.cpu().numpy()

        for p, f in zip(pos_np[::2], fatigue_np[::2]):
            color = (255, int(255 * (1.0 - f)), int(255 * (1.0 - f)))
            cv2.circle(vis_frame, (int(p[0]), int(p[1])), 1, color, -1)

        cv2.imshow("SPS Dipole V-Formation Engine", vis_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    total_time = time.time() - start_time
    print(f"Execution Completed: {STEPS} frames in {total_time:.2f}s ({STEPS / total_time:.1f} FPS)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dipole_demo()
