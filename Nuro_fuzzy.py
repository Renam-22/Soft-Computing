"""
anfis_skfuzzy.py
A small ANFIS-style neuro-fuzzy example using scikit-fuzzy for membership functions.

- Gaussian MFs using skfuzzy.gaussmf
- Sugeno-type consequents (linear per rule)
- Hybrid training: LS for consequents, finite-difference gradient descent for antecedent params

Run:
    python anfis_skfuzzy.py
"""
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# ---------------- Synthetic dataset ----------------
def make_data(N=400, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(-2, 2, size=(N,1))
    x2 = rng.uniform(-2, 2, size=(N,1))
    X = np.hstack([x1, x2])
    # target: nonlinear in x1, linear in x2 (toy problem)
    y = np.sin(x1) + 0.5 * x2 + 0.05 * rng.randn(N,1)
    return X.astype(np.float64), y.ravel().astype(np.float64)

# ---------------- ANFIS-style model using skfuzzy for MFs ----------------
class ANFIS_SKFuzzy:
    def __init__(self, D=2, M=2):
        """
        D = input dim, M = number of MFs per input (same for each input)
        R = M^D rules (cartesian product)
        """
        self.D = D
        self.M = M
        self.R = M ** D

        # Antecedent parameters: mu (D x M), sigma (D x M)
        mus = np.linspace(-1.5, 1.5, M)
        self.mu = np.tile(mus, (D,1)).astype(np.float64)    # shape (D,M)
        self.sigma = np.ones((D,M), dtype=np.float64) * 1.0 # initial width

        # Consequent parameters: for Sugeno linear: p_r (D params) + bias b_r -> (D+1) per rule
        self.theta = np.zeros((self.R, self.D + 1), dtype=np.float64)  # will be learned

        # Precompute rule index map: for each rule r, indices of MF for each input
        # Equivalent to cartesian product of [0..M-1] across D dims
        grids = np.array(np.meshgrid(*([np.arange(M)]*D), indexing='ij'))
        self.rule_map = grids.reshape(D, -1).T.astype(int)  # shape (R, D)

    def membership_matrix(self, X):
        """
        Compute membership values using skfuzzy.gaussmf
        X : (N, D)
        returns mem: shape (N, D, M)
        """
        N = X.shape[0]
        mem = np.zeros((N, self.D, self.M), dtype=np.float64)
        for d in range(self.D):
            for m in range(self.M):
                # Use scikit-fuzzy gaussian MF: fuzz.gaussmf(x, mean, sigma)
                mem[:, d, m] = fuzz.gaussmf(X[:, d], self.mu[d, m], self.sigma[d, m])
        return mem

    def rule_firing(self, membership):
        """
        membership: (N, D, M)
        returns w: (N, R) unnormalized firing strengths
        """
        N = membership.shape[0]
        R = self.R
        w = np.ones((N, R), dtype=np.float64)
        for r in range(R):
            for d in range(self.D):
                m_index = self.rule_map[r, d]
                w[:, r] *= membership[:, d, m_index]
        return w

    def predict(self, X):
        """
        Compute outputs with current parameters:
          - compute mem -> w -> normalized wbar
          - build regression Phi and compute y_pred = Phi * theta_vec
        returns (y_pred (N,), wbar (N,R), Phi (N, R*(D+1)))
        """
        membership = self.membership_matrix(X)
        w = self.rule_firing(membership)                  # (N,R)
        wsum = w.sum(axis=1, keepdims=True) + 1e-12
        wbar = w / wsum                                   # (N,R)

        N = X.shape[0]
        Phi = np.zeros((N, self.R * (self.D + 1)), dtype=np.float64)
        for r in range(self.R):
            # columns: wbar[:,r] * x_j  (for j in 0..D-1) and wbar[:,r] * 1
            for j in range(self.D):
                Phi[:, r*(self.D+1) + j] = wbar[:, r] * X[:, j]
            Phi[:, r*(self.D+1) + self.D] = wbar[:, r] * 1.0
        theta_vec = self.theta.reshape(-1)
        y_pred = Phi.dot(theta_vec)
        return y_pred, wbar, Phi

    def fit(self, X, y, epochs=30, lr=0.05, eps=1e-3, verbose=True):
        """
        Hybrid training:
          - For each epoch:
              1) compute membership & wbar
              2) solve linear least squares for consequent theta (Phi theta = y)
              3) compute mse
              4) compute numerical gradients of mse wrt mu and sigma via finite-difference
              5) update mu and sigma by gradient descent
        """
        N = X.shape[0]
        for ep in range(epochs):
            # 1) membership & normalized firing
            membership = self.membership_matrix(X)
            w = self.rule_firing(membership)
            wsum = w.sum(axis=1, keepdims=True) + 1e-12
            wbar = w / wsum

            # 2) form Phi and solve LS for theta
            Phi = np.zeros((N, self.R * (self.D + 1)), dtype=np.float64)
            for r in range(self.R):
                for j in range(self.D):
                    Phi[:, r*(self.D+1) + j] = wbar[:, r] * X[:, j]
                Phi[:, r*(self.D+1) + self.D] = wbar[:, r] * 1.0

            # least squares
            theta_vec, *_ = np.linalg.lstsq(Phi, y, rcond=None)
            self.theta = theta_vec.reshape(self.R, self.D + 1)

            # predictions and mse
            y_pred = Phi.dot(theta_vec)
            mse = np.mean((y - y_pred)**2)

            # 3) finite-difference gradient for mu and sigma
            mu_grad = np.zeros_like(self.mu)
            sig_grad = np.zeros_like(self.sigma)

            # We perturb each parameter slightly (+eps) to estimate gradient dMSE/dparam
            for d in range(self.D):
                for m in range(self.M):
                    # mu grad
                    orig = self.mu[d,m]
                    self.mu[d,m] = orig + eps
                    # recompute membership -> wbar -> LS -> mse+
                    mem_p = self.membership_matrix(X)
                    w_p = self.rule_firing(mem_p)
                    wsum_p = w_p.sum(axis=1, keepdims=True) + 1e-12
                    wbar_p = w_p / wsum_p
                    Phi_p = np.zeros((N, self.R * (self.D + 1)), dtype=np.float64)
                    for r in range(self.R):
                        for j in range(self.D):
                            Phi_p[:, r*(self.D+1) + j] = wbar_p[:, r] * X[:, j]
                        Phi_p[:, r*(self.D+1) + self.D] = wbar_p[:, r]
                    theta_p, *_ = np.linalg.lstsq(Phi_p, y, rcond=None)
                    y_p = Phi_p.dot(theta_p)
                    mse_p = np.mean((y - y_p)**2)
                    mu_grad[d,m] = (mse_p - mse) / eps
                    self.mu[d,m] = orig

                    # sigma grad
                    orig_s = self.sigma[d,m]
                    self.sigma[d,m] = orig_s + eps
                    mem_p = self.membership_matrix(X)
                    w_p = self.rule_firing(mem_p)
                    wsum_p = w_p.sum(axis=1, keepdims=True) + 1e-12
                    wbar_p = w_p / wsum_p
                    Phi_p = np.zeros((N, self.R * (self.D + 1)), dtype=np.float64)
                    for r in range(self.R):
                        for j in range(self.D):
                            Phi_p[:, r*(self.D+1) + j] = wbar_p[:, r] * X[:, j]
                        Phi_p[:, r*(self.D+1) + self.D] = wbar_p[:, r]
                    theta_p, *_ = np.linalg.lstsq(Phi_p, y, rcond=None)
                    y_p = Phi_p.dot(theta_p)
                    mse_p = np.mean((y - y_p)**2)
                    sig_grad[d,m] = (mse_p - mse) / eps
                    self.sigma[d,m] = orig_s

            # 4) gradient descent update (move opposite to gradient)
            self.mu -= lr * mu_grad
            # ensure sigma stays positive; use small lower bound
            self.sigma -= lr * sig_grad
            self.sigma = np.maximum(self.sigma, 1e-3)

            if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
                print(f"Epoch {ep+1}/{epochs}   MSE = {mse:.6e}")

        return

# ---------------- Demo and run ----------------
if __name__ == "__main__":
    X, y = make_data(500, seed=2)
    n_train = int(0.8 * X.shape[0])
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    model = ANFIS_SKFuzzy(D=2, M=2)   # 2 inputs, 2 MFs each -> 4 rules
    print("Initial mu:\n", model.mu)
    print("Initial sigma:\n", model.sigma)

    model.fit(Xtr, ytr, epochs=40, lr=0.1, eps=1e-3, verbose=True)

    y_pred_tr, _, _ = model.predict(Xtr)
    y_pred_te, _, _ = model.predict(Xte)
    mse_tr = np.mean((ytr - y_pred_tr)**2)
    mse_te = np.mean((yte - y_pred_te)**2)

    print("\nTraining MSE:", mse_tr)
    print("Test MSE:", mse_te)
    print("\nLearned mu:\n", model.mu)
    print("Learned sigma:\n", model.sigma)
    print("Learned consequents (theta) per rule [p1,p2,b]:\n", model.theta)

    # Quick scatter plot: true vs predicted on test
    plt.figure(figsize=(6,5))
    plt.scatter(yte, y_pred_te, alpha=0.4)
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("ANFIS (scikit-fuzzy MFs) - True vs Pred (test)")
    plt.grid(True)
    plt.show()
