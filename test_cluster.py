import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# configuration
WIDTH, HEIGHT = 256, 256
NUM_AGENTS = 2000
STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sps:
    def __init__(self, w: int, h: int, device: torch.device):
        self.w = w
        self.h = h
        self.device = device

        # Layer 0: Fast Detection / Trail Field
        # Shape: [1, 1, H, W] for torch.conv2d compatibility
        self.E_raw = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
        self.E_act = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)

        # Physics Parameters (Section 4.1 E_fast specifications)
        self.D = 0.15  # Diffusion rate
        self.decay = 0.92  # Dissipation (λ)
        self.dt = 1.0

        # Discrete 5-point Laplacian Stencil Matrix
        self.laplacian_kernel = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=device,
            )
            .view(1, 1, 3, 3)
        )

    def step_physics(self, injection_grid: torch.Tensor):
        """Executes diffusion, decay, and injection pipeline (Section 2.2)."""
        # 1. Diffusion via Conv2D stencil
        laplacian = F.conv2d(self.E_act, self.laplacian_kernel, padding=1)
        E_diffused = self.E_act + self.D * laplacian * self.dt

        # 2. Dissipation & Injection
        E_dissipated = E_diffused * self.decay
        self.E_raw = E_dissipated + injection_grid

        # 3. Activation & Clamping
        self.E_act = torch.clamp(self.E_raw, 0.0, 1.0)

    def sample_field_and_gradient(self, pos: torch.Tensor):
        """Bilinear read of pre-activation values and central-difference gradients (Section 3.1)."""
        # Normalize continuous positions to [-1, 1] range for grid_sample
        norm_x = (pos[:, 0] / (self.w - 1)) * 2.0 - 1.0
        norm_y = (pos[:, 1] / (self.h - 1)) * 2.0 - 1.0
        grid_coords = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)

        # 1. Bilinear Interpolation of Values (v_k)
        val = F.grid_sample(
            self.E_raw, grid_coords, align_corners=True, mode="bilinear"
        ).view(-1)

        # 2. Pre-activation Spatial Gradients (∇E_k)
        grad_x_field = (
            torch.roll(self.E_raw, shifts=-1, dims=3)
            - torch.roll(self.E_raw, shifts=1, dims=3)
        ) / 2.0
        grad_y_field = (
            torch.roll(self.E_raw, shifts=-1, dims=2)
            - torch.roll(self.E_raw, shifts=1, dims=2)
        ) / 2.0

        gx = F.grid_sample(
            grad_x_field, grid_coords, align_corners=True, mode="bilinear"
        ).view(-1)
        gy = F.grid_sample(
            grad_y_field, grid_coords, align_corners=True, mode="bilinear"
        ).view(-1)

        return val, torch.stack([gx, gy], dim=-1)


class AgentSwarm:
    def __init__(self, num_agents: int, w: int, h: int, device: torch.device):
        self.num_agents = num_agents
        self.w = w
        self.h = h
        self.device = device

        # Continuous positions (x, y)
        self.pos = torch.rand((num_agents, 2), device=device) * torch.tensor(
            [w - 1, h - 1], device=device
        )
        # Random initial headings (angles)
        self.angles = torch.rand(num_agents, device=device) * 2 * np.pi
        self.speed = 1.2
        self.injection_weight = 0.6

    def update_and_inject(self, sps: Sps):
        # 1. Read Field (Temporal Separation: Read pre-activation state before injecting)
        vals, grads = sps.sample_field_and_gradient(self.pos)

        # 2. Gradient Following Policy (Steer towards higher field concentrations)
        grad_mag = torch.norm(grads, dim=-1)
        target_angles = torch.atan2(grads[:, 1], grads[:, 0])

        # Epsilon-gradient check: Steer along gradient if signal exists, else apply small random wander
        has_signal = grad_mag > 1e-4
        rand_turn = (torch.rand(self.num_agents, device=self.device) - 0.5) * 0.4

        self.angles = torch.where(
            has_signal,
            # Smooth steering interpolation towards gradient angle
            self.angles + 0.3 * torch.atan2(
                torch.sin(target_angles - self.angles),
                torch.cos(target_angles - self.angles),
            ),
            self.angles + rand_turn,
        )

        # 3. Kinematic Step
        dx = torch.cos(self.angles) * self.speed
        dy = torch.sin(self.angles) * self.speed
        self.pos[:, 0] = torch.clamp(self.pos[:, 0] + dx, 0, self.w - 1.001)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1] + dy, 0, self.h - 1.001)

        # 4. Bilinear Injection Splatting across 4 nearest grid cells (Section 2.3)
        ix = torch.floor(self.pos[:, 0]).long()
        iy = torch.floor(self.pos[:, 1]).long()
        dx_val = self.pos[:, 0] - ix.float()
        dy_val = self.pos[:, 1] - iy.float()

        inj_grid = torch.zeros((self.h, self.w), device=self.device)

        # Flattened indices for atomic accumulation
        w_len = self.w
        w00 = (1.0 - dx_val) * (1.0 - dy_val) * self.injection_weight
        w10 = dx_val * (1.0 - dy_val) * self.injection_weight
        w01 = (1.0 - dx_val) * dy_val * self.injection_weight
        w11 = dx_val * dy_val * self.injection_weight

        inj_grid.put_(
            iy * w_len + ix, w00, accumulate=True
        )
        inj_grid.put_(
            iy * w_len + (ix + 1), w10, accumulate=True
        )
        inj_grid.put_(
            (iy + 1) * w_len + ix, w01, accumulate=True
        )
        inj_grid.put_(
            (iy + 1) * w_len + (ix + 1), w11, accumulate=True
        )

        return inj_grid.view(1, 1, self.h, self.w)


def run_demo():
    print(f"Initializing SPS Engine on Execution Device: {DEVICE}")
    sps = Sps(WIDTH, HEIGHT, DEVICE)
    swarm = AgentSwarm(NUM_AGENTS, WIDTH, HEIGHT, DEVICE)

    cv2.namedWindow("Substrate Perception System (SPS) Engine", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Substrate Perception System (SPS) Engine", 768, 768)

    start_time = time.time()
    for frame in range(STEPS):
        # Step A: Agent Perception, Movement, and Injection Splatting
        inj_grid = swarm.update_and_inject(sps)

        # Step B: Scalar Field Physics (Diffusion + Decay)
        sps.step_physics(inj_grid)

        # Step C: Visualization Engine (Field Intensity Map + Agent Overlay)
        field_np = sps.E_act.squeeze().cpu().numpy()
        vis_frame = cv2.applyColorMap((field_np * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # Render Agent positions (Red Dots)
        pos_np = swarm.pos.cpu().numpy()
        for p in pos_np[::3]:  # Render subset for performance
            cv2.circle(vis_frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)

        cv2.imshow("Substrate Perception System (SPS) Engine", vis_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    total_time = time.time() - start_time
    print(f"Execution Completed: {STEPS} frames in {total_time:.2f}s ({STEPS / total_time:.1f} FPS)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
