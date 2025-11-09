import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Rosenbrock Function (2D)
def rosenbrock(X):
    x, y = X
    return (1 - x)**2 + 100 * (y - x**2)**2

# Grey Wolf Optimizer
class GWO:
    def __init__(self, obj_func, lb, ub, dim=2, n_wolves=20, max_iter=60, r1_scale=1.0, r2_scale=1.0):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.r1_scale = r1_scale
        self.r2_scale = r2_scale

        self.positions = np.random.uniform(low=lb, high=ub, size=(n_wolves, dim))
        self.history = []
        self.best_fitness_history = []

    def optimize(self):
        for t in range(self.max_iter):
            fitness = np.apply_along_axis(self.obj_func, 1, self.positions)
            sorted_idx = np.argsort(fitness)
            alpha = self.positions[sorted_idx[0]]
            beta = self.positions[sorted_idx[1]]
            delta = self.positions[sorted_idx[2]]

            # record best fitness
            self.best_fitness_history.append(fitness[sorted_idx[0]])

            a = 2 - 2 * (t / self.max_iter)  # decreasing parameter
            new_positions = []
            for i in range(self.n_wolves):
                X = self.positions[i]
                for leader in [alpha, beta, delta]:
                    r1, r2 = np.random.rand() * self.r1_scale, np.random.rand() * self.r2_scale
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader - X)
                    X = leader - A * D
                X = np.clip(X, self.lb, self.ub)
                new_positions.append(X)
            self.positions = np.array(new_positions)
            self.history.append(self.positions.copy())
        return alpha

# --- Visualization in 3D ---
def animate_gwo_3d_userinput(max_iter=60, r1_scale=1.0, r2_scale=1.0):
    gwo = GWO(
        obj_func=rosenbrock, 
        lb=[-3, -2], ub=[3, 3], 
        n_wolves=20, 
        max_iter=max_iter,
        r1_scale=r1_scale,
        r2_scale=r2_scale
    )
    gwo.optimize()
    history = gwo.history

    # Prepare Rosenbrock surface
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-2, 3, 300)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax_curve = fig.add_subplot(122)

    # Plot Rosenbrock surface
    ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    ax3d.set_title("Grey Wolf Optimization on Rosenbrock (3D)", fontsize=12, fontweight='bold')
    ax3d.set_xlabel("X1", fontsize=10)
    ax3d.set_ylabel("X2", fontsize=10)
    ax3d.set_zlabel("f(X)", fontsize=10)

    # Wolves scatter (different markers for leaders)
    scat_wolves = ax3d.scatter([], [], [], c='white', s=40, edgecolors='black', alpha=0.8, label='Wolves')
    scat_alpha = ax3d.scatter([], [], [], c='red', s=120, marker='*', label='Alpha (Best)')
    scat_beta = ax3d.scatter([], [], [], c='orange', s=80, marker='D', label='Beta')
    scat_delta = ax3d.scatter([], [], [], c='green', s=80, marker='s', label='Delta')
    scat_prey = ax3d.scatter([1], [1], [0], c='black', marker='X', s=150, label='Prey (Optimum)')

    ax3d.legend(loc='upper left')

    # Convergence curve subplot
    ax_curve.set_title("Convergence Curve", fontsize=12, fontweight="bold")
    ax_curve.set_xlabel("Iteration", fontsize=10)
    ax_curve.set_ylabel("Best Fitness", fontsize=10)
    line_curve, = ax_curve.plot([], [], c="red", lw=2)
    ax_curve.grid(True, linestyle="--", alpha=0.6)

    def update(frame):
        wolves = history[frame]
        fitness = np.apply_along_axis(rosenbrock, 1, wolves)
        sorted_idx = np.argsort(fitness)

        alpha, beta, delta = wolves[sorted_idx[0]], wolves[sorted_idx[1]], wolves[sorted_idx[2]]

        Z_wolves = [rosenbrock(w) for w in wolves]
        Z_alpha, Z_beta, Z_delta = rosenbrock(alpha), rosenbrock(beta), rosenbrock(delta)

        # Update wolves
        scat_wolves._offsets3d = (wolves[:, 0], wolves[:, 1], Z_wolves)
        scat_alpha._offsets3d = ([alpha[0]], [alpha[1]], [Z_alpha])
        scat_beta._offsets3d = ([beta[0]], [beta[1]], [Z_beta])
        scat_delta._offsets3d = ([delta[0]], [delta[1]], [Z_delta])

        # Update convergence curve
        line_curve.set_data(range(frame+1), gwo.best_fitness_history[:frame+1])
        ax_curve.relim()
        ax_curve.autoscale_view()

        # Rotate camera for cinematic effect
        ax3d.view_init(elev=30, azim=30 + frame*2)

        return scat_wolves, scat_alpha, scat_beta, scat_delta, scat_prey, line_curve

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=250, repeat=False)
    plt.tight_layout()
    plt.show()

# Run with user input
if __name__ == "__main__":
    max_iter = int(input("Enter number of iterations (e.g. 60, 100, 200): "))
    r1_scale = float(input("Enter randomness scale for r1 (0.1 - 1.0): "))
    r2_scale = float(input("Enter randomness scale for r2 (0.1 - 1.0): "))

    animate_gwo_3d_userinput(max_iter=max_iter, r1_scale=r1_scale, r2_scale=r2_scale)
