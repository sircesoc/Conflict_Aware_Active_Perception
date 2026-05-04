# Conflict-Aware CBF for Unicycle Dynamics

Safe navigation with active perception in 3D Gaussian Splatting fields using Control Barrier Functions.

## Installation

```bash
# create and activate a conda environment (Python 3.9–3.12)
conda create -n cap_cbf python=3.11 -y
conda activate cap_cbf

# install dependencies
pip install -r requirements.txt
```

## Running

### Single formulation

```bash
# h_p formulation  —  single perception barrier h_p = I - I_c
python run_conflict_cbf.py params/yq_large_cloud.mat \
    --formulation eq32 --dt 0.05 --i-c 0.7 \
    --start -1.8 -1.8 --goal 6 -11

# h_pi + h_eta formulation  —  two perception barriers (spatial + angular)
python run_conflict_cbf.py params/yq_large_cloud.mat \
    --formulation eq2 --dt 0.05 --phi-max 60 \
    --start -1.8 -1.8 --goal 6 -11
```

### Compare formulations side-by-side

Runs both on the same scene/start/goal and produces overlay plots + metrics:

```bash
python compare_formulations.py params/yq_large_cloud.mat \
    --dt 0.05 --i-c 0.7 --k-slack 0.7 --w2-slack 0.10 --w-track 1 \
    --occupancy-scale 3 --z-thickness 0.4 \
    --fov-deg 60 --obs-decay 0.003 --save-gif \
    --start -1.8 -1.8 --goal 6 -11
```

### Run only one formulation in compare mode

```bash
python compare_formulations.py params/yq_large_cloud.mat \
    --dt 0.05 --i-c 0.7 --k-slack 0.7 --w-track 1 \
    --occupancy-scale 3 --z-thickness 0.4 \
    --fov-deg 60 --obs-decay 0.003 --save-gif \
    --formulations eq32 \
    --start -1.8 -1.8 --goal 6 -11
```

### Sweep slack weights

Run the same formulation with different perception slack weights to study the safety-vs-perception trade-off:

```bash
python compare_formulations.py params/yq_large_cloud.mat \
    --dt 0.05 --adaptive-bias-target 0.005 \
    --i-c 1.4 --w2-slack 0.10 --w-track 1 \
    --occupancy-scale 3 --z-thickness 0.4 \
    --fov-deg 60 --obs-decay 0.003 --save-gif \
    --formulations eq32 \
    --start 1.0 -9.0 --goal 2.2 0 \
    --k-slack-sweep 0.1 1.0 \
    --safety-lookahead 0.15
```

### Interactive mode

Click start and goal on the occupancy grid instead of passing coordinates:

```bash
python compare_formulations.py params/yq_large_cloud.mat --interactive --save-gif
```

## Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--formulation` | `conflict` | Controller: `eq32` (h_p), `eq2` (h_pi + h_eta), or `conflict` (legacy) |
| `--dt` | `0.01` | Simulation timestep (s). Use `0.05` for faster runs |
| `--start X Y` | — | Start position in world coordinates |
| `--goal X Y` | — | Goal position |
| `--interactive` | off | Click start/goal on the grid |
| `--i-c` | `0.1` | EIG threshold $I_c$ for `eq32` (higher = more cautious perception) |
| `--phi-max` | `45` | Heading FOV half-angle (deg) for `eq2` |
| `--k-slack` | `0.3` | Perception slack weight (higher = softer perception constraint) |
| `--w2-slack` | `0.1` | Angular slack weight, `eq2` only |
| `--w-track` | `10` | Tracking cost weight in QP |
| `--safety-lookahead` | `0.0` | Evaluate safety barrier this far ahead along heading (m) |
| `--occupancy-scale` | `1.0` | Multiplier on occupancy radii from uncertainty |
| `--z-thickness` | `0.5` | Height band (m) for the 2D occupancy slice |
| `--obs-decay` | `0.0` | Per-step uncertainty decay for Gaussians in FOV |
| `--fov-deg` | `90` | Sensor FOV for observation simulation (deg) |
| `--save-gif` | off | Save animated trajectory GIF |
| `--formulations` | `eq32 eq2` | (compare only) Which formulations to run |
| `--k-slack-sweep W ...` | — | (compare only) Run each formulation once per slack weight |

## Outputs

- `compare_formulations.png` — trajectory overlay on the 3DGS scene
- `compare_formulations_metrics.png` — multi-panel metrics (safety, perception, smoothness)
- `compare_formulations.gif` — animated trajectory rollout (with `--save-gif`)
