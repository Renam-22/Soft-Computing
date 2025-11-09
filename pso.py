import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# All parameters
# -----------------------
SWARM_SIZE = 8
DIM = 2
LOW, HIGH = -5.0, 5.0
MAX_ITERS = 20
W = 0.7
C1 = 1.5
C2 = 1.5
VEL_CLIP = (HIGH - LOW) * 0.5
CSV_FILE = "pso_positions.csv"
GRAPH_DIR = "graph"     
TRAJ_FILE = os.path.join(GRAPH_DIR, "trajectories.png")

random.seed(42)
np.random.seed(42)

# create graph folder
os.makedirs(GRAPH_DIR, exist_ok=True)

# -----------------------
# Rosenbrock function
# -----------------------
def rosenbrock(x):
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

# -----------------------
# Initialization
# -----------------------
pos = np.random.uniform(LOW, HIGH, size=(SWARM_SIZE, DIM))
vel = np.random.uniform(-0.1, 0.1, size=(SWARM_SIZE, DIM)) * (HIGH - LOW)
pbest_pos = pos.copy()
pbest_val = np.array([rosenbrock(p) for p in pos])
gbest_idx = int(np.argmin(pbest_val))
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]

# prepare CSV: write header
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "gen", "particle",
        "x1", "x2",
        "v1", "v2",
        "pbest_x1", "pbest_x2", "pbest_val",
        "gbest_x1", "gbest_x2", "gbest_val"
    ])

# store trajectories: generation 0..MAX_ITERS
trajectories = np.zeros((MAX_ITERS + 1, SWARM_SIZE, DIM))
trajectories[0] = pos.copy()

# history for convergence plot
gen_history = []
gbest_history = []

print("Running PSO and saving generation plots to the 'graph/' folder...")

for gen in range(MAX_ITERS):
    # evaluate current fitness
    fitness = np.array([rosenbrock(p) for p in pos])

    # update personal bests
    improved = fitness < pbest_val
    if improved.any():
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fitness[improved]

    # update global best
    current_best_idx = int(np.argmin(pbest_val))
    if pbest_val[current_best_idx] < gbest_val:
        gbest_val = pbest_val[current_best_idx]
        gbest_pos = pbest_pos[current_best_idx].copy()

    # record history
    gen_history.append(gen)
    gbest_history.append(gbest_val)

    # write CSV rows for this generation
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for pid in range(SWARM_SIZE):
            writer.writerow([
                gen, pid,
                float(pos[pid,0]), float(pos[pid,1]),
                float(vel[pid,0]), float(vel[pid,1]),
                float(pbest_pos[pid,0]), float(pbest_pos[pid,1]), float(pbest_val[pid]),
                float(gbest_pos[0]), float(gbest_pos[1]), float(gbest_val)
            ])

    # --- create and save the per-generation plot (scatter + convergence) ---
    fig, (ax_scatter, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))

    # scatter subplot
    ax_scatter.set_xlim(LOW - 0.5, HIGH + 0.5)
    ax_scatter.set_ylim(LOW - 0.5, HIGH + 0.5)
    ax_scatter.set_xlabel("x1"); ax_scatter.set_ylabel("x2")
    ax_scatter.set_title(f"PSO (gen {gen:03d}) - gbest {gbest_val:.3e}")
    ax_scatter.grid(True, linestyle=":", alpha=0.6)
    ax_scatter.scatter(pos[:,0], pos[:,1], s=60, c='red', label='particles')
    ax_scatter.scatter(pbest_pos[:,0], pbest_pos[:,1], s=25, c='orange', marker='x', alpha=0.8, label='pbest')
    ax_scatter.scatter([gbest_pos[0]], [gbest_pos[1]], c='blue', s=120, marker='*', label='gbest')
    ax_scatter.legend(loc='upper left')

    # convergence subplot (log scale)
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Global best value (log scale)")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, linestyle=":", alpha=0.6)
    ax_conv.semilogy(gen_history, gbest_history, marker='o', linestyle='-')
    ax_conv.scatter([gen_history[-1]], [gbest_history[-1]], color='green', s=50)
    ax_conv.set_xlim(-0.5, MAX_ITERS + 0.5)

    fig.tight_layout()
    path = os.path.join(GRAPH_DIR, f"gen_{gen:03d}.png")
    fig.savefig(path)
    plt.close(fig)

    # ---------------------
    # PSO velocity & position update
    # ---------------------
    r1 = np.random.rand(SWARM_SIZE, DIM)
    r2 = np.random.rand(SWARM_SIZE, DIM)
    cognitive = C1 * r1 * (pbest_pos - pos)
    social = C2 * r2 * (gbest_pos - pos)
    vel = W * vel + cognitive + social
    vel = np.clip(vel, -VEL_CLIP, VEL_CLIP)
    pos = pos + vel
    pos = np.clip(pos, LOW, HIGH)

    # store trajectories after the move
    trajectories[gen + 1] = pos.copy()

    # optional console feedback
    if gen % 5 == 0:
        print(f"Gen {gen:3d} saved -> {path} | gbest_val = {gbest_val:.6e}")

# after all generations: save trajectories plot and display it
fig, ax = plt.subplots(figsize=(7,6))
colors = plt.cm.get_cmap("tab10", SWARM_SIZE)
for pid in range(SWARM_SIZE):
    traj = trajectories[:, pid, :]
    ax.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, label=f"p{pid}", color=colors(pid))
    ax.scatter([traj[0,0]], [traj[0,1]], color=colors(pid), marker='s', s=30, alpha=0.7)  # start
    ax.scatter([traj[-1,0]], [traj[-1,1]], color=colors(pid), marker='*', s=70, edgecolor='k')  # end

ax.set_xlim(LOW - 0.5, HIGH + 0.5)
ax.set_ylim(LOW - 0.5, HIGH + 0.5)
ax.set_title("Particle trajectories over generations")
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(TRAJ_FILE)
print(f"\nTrajectories plot saved to: {TRAJ_FILE}")
print("Displaying trajectories plot now...")
plt.show()   # show only this final trajectories plot

print("\nPSO finished.")
print(f"Final global best value: {gbest_val:.10e}")
print(f"Final global best position: x1={gbest_pos[0]:.6f}, x2={gbest_pos[1]:.6f}")
print(f"CSV saved as: {CSV_FILE}")
print(f"All per-generation plots are in the '{GRAPH_DIR}/' folder.")