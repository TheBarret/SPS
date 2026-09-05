# Concept Idea: Substrate Perception System (SPS)

A Substrate Perception System (SPS) is a field-based multi-agent perception architecture  
that replaces geometric detection  (raycasts, radius checks, spatial partitioning) with passive scalar field interactions.   
Agents perceive their environment by reading local values and gradients from a set of dissipating,  
diffusing energy fields—achieving O(1) perception per agent regardless of total agent count.  

**Detection Mechanism:**  
Detection is not an active query but a passive byproduct of existing within a shared computational substrate.  

**Computation:**  
Total cost per frame is O(W·H·K + N), where W×H is grid resolution, K is the number of field layers,  
and N is the number of agents. This represents a favorable trade-off compared to O(N²) pairwise systems when N ≫ W·H,  
but the field update term dominates at high resolutions.  
The system is therefore best suited for scenarios with dense agent populations and moderate spatial precision requirements.  

---

## 2. Pipeline

### 2.1 Field Representation

Let the environment be represented as a discrete 2D grid of dimensions `W × H`.  
The system maintains a stack of `K` scalar fields:  

```
E_k ∈ ℝ^(W×H), k ∈ {0, 1, ..., K-1}
```

Each field is bounded to the range `[0.0, 1.0]` at all times.  

**Field Parameters:**

| Parameter | Symbol | Type | Description |
| :--- | :--- | :--- | :--- |
| Diffusion Rate | `D_k` | Float | Diffusion coefficient (spread per timestep). |
| Dissipation Rate | `λ_k` | Float | Multiplicative decay factor per timestep (0.0 < λ < 1.0). |
| Activation Function | `f_k: ℝ → ℝ` | Function | Nonlinear transform applied post-physics. |
| Injection Weight | `ω_k` | Float | Energy contribution per agent action. |
| Detection Threshold | `τ_k` | Float | Effective signal-to-noise cutoff for this layer. |

### 2.2 Field Update Equation (Single Timestep)

For each field `k`, the update from time `t` to `t+1`  
is implemented via iterative local-stencil diffusion (discretized heat equation):  

```
∇²E_k = E_k[x-1,y] + E_k[x+1,y] + E_k[x,y-1] + E_k[x,y+1] - 4·E_k[x,y]
E_diffused = E_k + D_k · ∇²E_k · dt
E_dissipated = E_diffused · λ_k
E_injected = E_dissipated + I_k^(t)
E_k^(t+1) = clamp( f_k(E_injected), 0.0, 1.0 )
```

Where:

- `D_k` is the diffusion coefficient (typically 0.1–0.3 for stable integration).
- `dt` is the timestep (typically 1.0 for discrete frame stepping).
- `I_k^(t)` is the sparse injection matrix from all agents at time `t`.
- `clamp(x, 0, 1)` ensures values remain within the valid range.

**Critical Ordering Note:**   
Gradients and temporal derivatives MUST be computed from the pre-activation field to preserve directional information:  
```
E_raw = E_injected  // Pre-activation
∇E = compute_gradient(E_raw)                   // Directional information preserved
E_activated = clamp(f_k(E_raw), 0.0, 1.0)     // Post-activation for perception
```

**Rationale for Local-Stencil Diffusion:**  

- Approximately 9 operations per pixel per layer per frame.
- Spread naturally grows over multiple frames (physical diffusion).
- No large kernel convolution required (O(W·H) instead of O(W·H·kernel_size²)).
- Well-established in reaction-diffusion and slime mold simulation literature.
- Can be accelerated via GPU or SIMD vectorization.

### 2.3 Injection Splatting

Agents inject energy at continuous positions `(x, y) ∈ ℝ²`.  
To preserve smooth gradients and avoid grid-snap artifacts,  
injection is bilinearly splatted across the four nearest cells:   

```
ix = floor(x), iy = floor(y)
dx = x - ix, dy = y - dy

I_k[ix,   iy]   += ω_k · injection_amount · (1-dx) · (1-dy)
I_k[ix+1, iy]   += ω_k · injection_amount · dx · (1-dy)
I_k[ix,   iy+1] += ω_k · injection_amount · (1-dx) · dy
I_k[ix+1, iy+1] += ω_k · injection_amount · dx · dy
```

This mirrors the bilinear read operation, ensuring symmetry and preventing directional bias.  

---

## 3. Perception Vector Construction

### 3.1 Agent Read Protocol

At each timestep, an agent at continuous position `(x, y)` constructs a perception vector `P` via:  

```
1. Read pre-activation value via bilinear interpolation:
   v_k = bilinear(E_raw, x, y)

2. Compute spatial gradient via bilinear interpolation of finite differences:
   ∇E_k = bilinear( (E_raw[x+1] - E_raw[x-1]) / 2,
                    (E_raw[y+1] - E_raw[y-1]) / 2 )

3. Compute temporal derivative (requires history):
   dv_k = v_k - v_k_prev

4. Optional: Read local neighborhood patch (for encirclement detection):
   N_k = bilinear_patch(E_raw, x, y, radius=1)

5. Package perception vector:
   P = [v_0, ∇E_0, dv_0, v_1, ∇E_1, dv_1, ...]
```

**Implementation Notes:**

**Boundary handling:**  
Use absorption (field value tapers to zero near edges) rather than reflective padding for trail layers,  
to prevent phantom mirror trails.  

**Pre-activation vs. Post-activation:**  
Gradients are computed from `E_raw` to prevent saturation-induced flattening.  

**Temporal derivative:**  
Requires storing previous frame's field state or per-agent history buffer.  

### 3.2 Alternative: Learned Perception Features

Instead of hand-engineering the perception vector (gradient, temporal derivative, neighborhood),  
the system MAY use a small learned convolutional module that processes the raw local patch directly:  

```
P_k = small_CNN(E_raw_patch, parameters_θ_k)
P = concat(P_0, P_1, ..., P_{K-1})
```

This approach:

- Collapses gradient computation, encirclement detection, and feature extraction into one module. 
- Can learn features not anticipated by hand-design (e.g., split trails, curved trajectories).
- Aligns with Neural Cellular Automata literature (Mordvintsev et al., 2020).
- **Trade-off:** Requires training; less interpretable; adds computational cost.

**Default Implementation:**  
Hand-engineered vector (Section 3.1) for initial deployment. Learned module available as extension (Section 11).  

---

## 4. Activation Functions

### 4.1 Default Stack (K=3)

| Layer | D_k | λ_k | f_k(x) | τ_k | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E_fast** | 0.15 | 0.95 | `max(0, min(x, 1))` | 0.30 | Real-time detection. Sharp, short-lived peaks. |
| **E_medium** | 0.25 | 0.85 | `sigmoid(3.0*(x-0.15)) * (1 + 0.5*H)` | 0.20 | Trail following. Smooth gradients. |
| **E_slow** | 0.40 | 0.70 | `x * exp(-0.1 * t^0.5)` | 0.10 | Long-term memory. Persistent territory marks. |

### 4.2 Trail-Sigmoid with Hysteresis

```
f_trail(x, H) = sigmoid(G · (x - T)) · (1 + 0.5 · H)
```
Where:

- `G` = Gain (amplification factor, default 3.0).
- `T` = Threshold (noise floor, default 0.15).
- `H` = Per-node hysteresis memory, updated as:
  ```
  H_new = H_old · 0.9 + (x > 0.3 ? 1.0 : 0.0)
  ```

**Behavior:**
- Suppresses noise below `T`.
- Amplifies medium signals to visible levels.
- Maintains trail persistence after agent departure via hysteresis.

### 4.3 Taper (Stretched Exponential)

```
f_taper(x, t) = x · exp(-α · t^β)
```
Where:

- `α` = Decay rate (default 0.1).
- `β` = Stretch exponent (default 0.5).
- `t` = Elapsed time since last injection.

**Behavior:**
- Rapid initial decay followed by long, slow tail.
- Prevents full extinction, creating geological memory layers.

### 4.4 Calibration: Effective Detection Radius

Given the field parameters, the steady-state falloff profile of a point source injected at rate `ω` with diffusion `D`,   dissipation `λ`, and threshold `τ` can be approximated.  
The effective detection radius `R_eff` is the radius at which the steady-state value drops below `τ`.  

For the default E_fast layer (D=0.15, λ=0.95, ω=0.5, τ=0.30), the steady-state profile follows approximately:  

```
E(r) ≈ (ω / (2π · D · (1-λ))) · exp(-r · sqrt((1-λ)/D))
```

Solving for `E(R_eff) = τ` yields the effective detection radius.  
This calibration is REQUIRED for validating detection accuracy metrics against geometric baseline systems.  

---

## 5. Gradient Computation & Directional Handling

### 5.1 Standard Central Difference

For field `E` at grid position `(x, y)`:

```
∇E(x,y) = ( (E[x+1,y] - E[x-1,y]) / 2,
            (E[x,y+1] - E[x,y-1]) / 2 )
```

For continuous positions, the gradient is bilinearly interpolated from the discrete gradient field.

**Limitation:** Opposite gradients cancel. Two agents on opposite sides produce `∇E ≈ (0,0)` despite high local energy.

### 5.2 Epsilon-Gradient Fallback

If `||∇E|| < ε` (where ε is a small threshold, e.g., `1e-6`):

```
if ||∇E|| < ε:
    // Use wider stencil to find distant peaks
    ∇E_wide = compute_gradient(E, radius=3)
    if ||∇E_wide|| > ε:
        ∇E = ∇E_wide
    else:
        // No signal: random walk, inertia, or explore
        action = random_walk() or maintain_heading()
```

**Purpose:** Prevents agents from freezing or drifting randomly in flat, dissipated regions.

### 5.3 8-Directional Neighborhood Vector (Optional)

To preserve directional information under conflicting signals, the perception vector MAY include the full 8-neighborhood:

```
N_k(x,y) = bilinear_patch(E, x, y, radius=1)
```

This vector encodes the full spatial structure around the agent, allowing policy networks to detect encirclement, clusters, and split trails. If using a learned perception module (Section 3.2), this is handled automatically.

---

## 6. Self-Signal Handling

### 6.1 The Nonlinear Entanglement Problem

A mathematically pure solution—subtracting each agent's own private "self field" from the total—is **not feasible** because:

1. Diffusion and decay are linear operations and commute with addition.
2. However, the activation function `f_k(x)` is nonlinear and applied every timestep to the *combined* field.
3. `f(a + b) ≠ f(a) + f(b)` after one timestep, the agent's own contribution is nonlinearly entangled with all others.
4. Separating them would require storing a full parallel grid per agent (O(N·W·H)), which destroys the O(N) scaling.

### 6.2 Practical Solutions

**Solution A: Temporal Separation (Default)**
```
1. read_perception(position)      // Uses E_raw from previous timestep
2. policy(P) → action
3. apply_action(action)
4. inject_energy(action)          // Updates E_raw for next timestep
```

This is simple, cheap, and prevents the agent from perceiving its own **current-frame** injection.

**Solution B: Short-Window Correction**
```
E_perceived = E_raw - expected_own_contribution(last_N_injections)
```
Where `expected_own_contribution` is computed from the agent's known recent positions, `ω_k`, and the linear decay curve (ignoring nonlinear activation). This only corrects for the last 1-3 timesteps; older self-trail residuals are accepted as useful signal.

**Solution C: Reframing Self-Trail as Feature**
An agent sensing its own week-old trail is often useful information (e.g., "I've already searched here"). For long-memory layers, self-trail is NOT considered a bug; it becomes part of the agent's internal map of explored territory. The temporal separation (Solution A) handles the immediate self-signal; longer-term self-trail is treated as environmental memory.

**Recommendation:** Implement Solution A as default. Implement Solution B as an optional refinement for E_fast layer only. Solution C is the default behavior for E_slow layer.

---

## 7. Grid Resolution & Spatial Aliasing

### 7.1 The Resolution Tradeoff

- **Low Resolution (e.g., 64×64):** Fast field updates, but two agents in the same cell become indistinguishable.
- **High Resolution (e.g., 1024×1024):** High spatial precision, but field update cost grows as O(W·H).

The total cost per frame is `O(W·H·K + N)`. The system is optimal when `N ≫ W·H` (dense agent populations). For sparse agents or high-resolution requirements, traditional spatial partitioning may be more efficient.

### 7.2 Bilinear Interpolation for Continuous Positions

For agents occupying continuous positions `(x, y) ∈ ℝ²`, read values via bilinear interpolation from the discrete grid:

```
E(x,y) = (1-dx)·(1-dy)·E[ix,iy]
       + dx·(1-dy)·E[ix+1,iy]
       + (1-dx)·dy·E[ix,iy+1]
       + dx·dy·E[ix+1,iy+1]
```
Where `(ix, iy) = floor(x, y)` and `(dx, dy) = (x-ix, y-iy)`.

**Note:** Injection splatting (Section 2.3) mirrors this operation exactly to maintain symmetry.

### 7.3 Hierarchical Field Architecture

| Level | Resolution | D_k | λ_k | Update Frequency | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fine** | Full (W×H) | 0.15 | 0.95 | Every frame | Precise short-range interactions. |
| **Medium** | W/2 × H/2 | 0.25 | 0.85 | Every 2 frames | Navigation and mid-range trails. |
| **Coarse** | W/4 × H/4 | 0.40 | 0.70 | Every 4 frames | Long-term memory and territory. |

Agents read all levels and combine into a unified perception vector. Coarse levels update less frequently to reduce computational load.

---

## 8. Agent Model

### 8.1 Agent Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `position` | `(x, y) ∈ ℝ²` | Continuous position in world space. |
| `signature` | `[ω_0, ω_1, ..., ω_{K-1}]` | Injection weights per field layer. |
| `injection_amount` | `Float` | Energy emitted per action (default 0.5). |
| `energy_budget` | `Float` | Maximum total injection per timestep (optional). |
| `policy` | `Function(P) → Action` | Decision-making module (NN, rule-based, etc.). |
| `history` | `List[v_k_prev]` | Previous frame values for temporal derivative. |

### 8.2 Agent Action Loop (Per Timestep)

```
1. Perception:
   P = read_perception(position, E_raw_fields)

2. Decision:
   action = policy(P)

3. Execution:
   apply_action(action)  // May include movement, state changes, etc.

4. Injection:
   if inject_energy(action):
       for each field k:
           splat_injection(E_k, position, ω_k · injection_amount)
```

### 8.3 Self-Signal Temporal Separation

**This is the default and recommended sequence.** It prevents the agent from perceiving its own newly injected energy.

---

## 9. Boundary Conditions

### 9.1 Choice of Boundary Condition

| Boundary Type | Behavior | Use Case | Issue |
| :--- | :--- | :--- | :--- |
| **Absorption** | Field value tapers to zero at edges. | Environments with hard walls. | Field intensity lower near boundaries. |
| **Periodic** | Field wraps around edges. | Toroidal worlds (e.g., strategy game maps). | Requires world topology to match. |
| **Reflective** | Field mirrors at edges. | Symmetric arenas. | **Phantom trails** near walls (agent trails mirrored). |

### 9.2 Recommendation

Use **absorption** for trail-following layers to prevent phantom mirror trails. Use **periodic** for worlds with wrap-around topology. Use **reflective** only for symmetric, closed arenas where mirroring is physically meaningful.

**Implementation:** For absorption, apply an edge fade multiplier:

```
E_k[edge_region] *= edge_fade_factor  // e.g., 0.99 near edges
```

---

## 10. Implementation Skeleton Idea (PyTorch)

```python
import torch
import torch.nn.functional as F

class FieldStack:
    def __init__(self, W, H, K, D, lambdas, activations, injection_weights):
        """
        W, H: Grid dimensions
        K: Number of field layers
        D: Diffusion coefficients per layer
        lambdas: Dissipation rates per layer
        activations: Activation functions per layer
        injection_weights: Agent injection weights per layer
        """
        self.W, self.H, self.K = W, H, K
        self.D = D
        self.lambdas = lambdas
        self.activations = activations
        self.injection_weights = injection_weights
        
        # Field storage: [K, H, W]
        self.fields = torch.zeros(K, H, W)
        self.raw_fields = torch.zeros(K, H, W)  # Pre-activation
        self.hysteresis = torch.zeros(K, H, W)  # For trail-sigmoid
        
        # Laplacian kernel: 5-point stencil
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def step(self, injection_matrix, dt=1.0):
        """
        injection_matrix: [K, H, W] sparse energy additions from agents
        """
        for k in range(self.K):
            # Compute Laplacian via convolution
            laplacian = F.conv2d(
                self.fields[k:k+1].unsqueeze(0),
                self.laplacian_kernel,
                padding=1
            ).squeeze()
            
            # Diffusion
            diffused = self.fields[k] + self.D[k] * laplacian * dt
            
            # Dissipation
            dissipated = diffused * self.lambdas[k]
            
            # Injection
            injected = dissipated + injection_matrix[k]
            
            # Store raw for gradient computation
            self.raw_fields[k] = injected.clone()
            
            # Activation
            activated = self.activations[k](injected)
            
            # Clamp
            self.fields[k] = torch.clamp(activated, 0.0, 1.0)

    def read_perception(self, x, y):
        """
        Bilinear interpolation of field values and gradients at position (x, y).
        Returns perception vector P.
        """
        # Implementation of bilinear interpolation for continuous positions
        # Returns [v_k, ∇E_k, dv_k] for each layer k
        pass

    def splat_injection(self, k, x, y, amount):
        """
        Bilinearly splat energy injection at continuous position (x, y).
        """
        ix, iy = int(x), int(y)
        dx, dy = x - ix, y - iy
        
        self.injection_buffer[k, ix, iy] += amount * (1-dx) * (1-dy)
        self.injection_buffer[k, ix+1, iy] += amount * dx * (1-dy)
        self.injection_buffer[k, ix, iy+1] += amount * (1-dx) * dy
        self.injection_buffer[k, ix+1, iy+1] += amount * dx * dy
```

---

## 11. Success Metrics & Validation

| Metric | Definition | Target |
| :--- | :--- | :--- |
| **Detection Accuracy** | TP/(TP+FN) for agents within R_eff. | > 95% |
| **False Positive Rate** | FP/(FP+TN) | < 5% |
| **Trail Following Efficiency** | Path length ratio (agent path / target path) | < 1.5× |
| **Scalability (CPU)** | FPS for N agents on single thread | 30 FPS at N=1000 |
| **Scalability (GPU)** | FPS for N agents on single GPU | 30 FPS at N=10000 |
| **Perception Latency** | Time per agent read step | < 1 ms |

**Calibration Note:** Detection accuracy metrics MUST be defined relative to the calibrated effective radius `R_eff` derived from field parameters (Section 4.4). Without this calibration, comparison to geometric baselines is invalid.

---

## 12. Known Limitations & Mitigations

| Limitation | Impact | Mitigation |
| :--- | :--- | :--- |
| **Grid Aliasing** | Two agents in same cell indistinguishable. | Bilinear interpolation + hierarchical fields. |
| **Gradient Collapse** | Saturated fields have zero gradient. | Compute gradients pre-activation; use epsilon fallback. |
| **Self-Signal Nonlinear Entanglement** | Cannot cleanly separate self from others. | Temporal separation + short-window correction + reframe as feature. |
| **Directional Cancellation** | Opposite gradients cancel. | Use learned perception module or 8-directional vector. |
| **Boundary Phantom Trails** | Reflective boundaries mirror trails. | Use absorption or periodic boundaries; avoid reflection for trails. |
| **High Resolution Cost** | Field update dominates at W×H > N². | Hierarchical fields; use lower resolution for long-range layers. |
| **Parameter Calibration** | Unknown relationship to detection radius. | Derive R_eff analytically for each layer (Section 4.4). |

---

## 13. Extension Points (Iteration 2+)

| Extension | Description |
| :--- | :--- |
| **Learnable Parameters** | Train D, λ, G, T, α, β via RL or gradient descent. |
| **Learned Perception Module** | Replace hand-engineered features with small CNN reading local patch. |
| **Dynamic Spectral Tuning** | Agents adjust injection weights based on context. |
| **Field-Based Communication** | Encode messages in injection patterns (amplitude or frequency modulation). |
| **Terrain Masks** | Static fields representing walls, obstacles, or terrain absorption coefficients. |
| **Temporal Convolution** | Replace simple dissipation with learned temporal filters. |
| **Multi-Agent Cooperation** | Agents amplify each other's trails (positive stigmergy). |
| **Active Stealth** | Agents inject negative energy (valleys) to hide or confuse. |

---

## 14. Reference Implementation Checklist

- Grid allocation (`W × H × K` floats) with initialization to `0.0`.
- Laplacian stencil (5-point) for diffusion.
- Dissipation step (multiplication by `λ_k`).
- Sparse injection via bilinear splatting.
- Activation functions (ReLU, Trail-Sigmoid with hysteresis, Taper).
- Pre-activation storage for gradient computation.
- Bilinear interpolation for continuous reads.
- Gradient computation (central difference, pre-activation).
- Epsilon-gradient fallback for flat regions.
- Self-signal temporal separation (read before inject).
- Agent policy interface (NN or rule-based).
- Boundary handling (absorption or periodic).
- Effective radius calibration (Section 4.4).
- Visualization (heatmaps with agent overlays).
- Logging and data export for analysis.
