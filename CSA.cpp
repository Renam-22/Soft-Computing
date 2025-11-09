// simple_cuckoo.cpp
// Simple, easy-to-read Cuckoo Search (2D) implementation
// Minimizes f(x,y) = x^2 + y^2 (you can replace the fitness function)

#include <bits/stdc++.h>
using namespace std;

// ----------------------- helpers -----------------------
double rand01() { return (double)rand() / RAND_MAX; }

// simple normal (mean 0, std 1) using Box-Muller
double randn() {
    static bool have = false;
    static double gset;
    if (have) { have = false; return gset; }
    double u1, u2, s;
    do {
        u1 = 2.0 * rand01() - 1.0;
        u2 = 2.0 * rand01() - 1.0;
        s = u1*u1 + u2*u2;
    } while (s == 0.0 || s >= 1.0);
    double mul = sqrt(-2.0 * log(s) / s);
    gset = u1 * mul;
    have = true;
    return u2 * mul;
}

// clamp value to [lo, hi]
double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ----------------------- Levy (Mantegna) -----------------------
// returns one scalar step drawn from a Levy distribution with exponent beta (1<beta<=2)
// Mantegna's algorithm
double levy_step(double beta) {
    // sigma_u
    double num = tgamma(1 + beta) * sin(M_PI * beta / 2.0);
    double den = tgamma((1 + beta) / 2.0) * beta * pow(2.0, (beta - 1.0) / 2.0);
    double sigma_u = pow(num / den, 1.0 / beta);

    double u = randn() * sigma_u;
    double v = randn();
    double step = u / pow(fabs(v), 1.0 / beta);
    return step;
}

// ----------------------- fitness -----------------------
// simple test fitness: sphere function (minimize)
// Replace with any problem-specific fitness if needed
double fitness(double x, double y) {
    return x*x + y*y;
}

// ----------------------- main CSA -----------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand((unsigned) time(nullptr));

    int nests, iterations;
    double xmin, xmax;
    cout << "Number of nests (population): "; if(!(cin >> nests) || nests < 1) return 0;
    cout << "Number of iterations: "; cin >> iterations;
    cout << "Search range min and max (example -10 10): "; cin >> xmin >> xmax;

    // CSA parameters (simple defaults)
    double alpha = 0.5;    // step size scaling
    double pa = 0.25;      // abandonment probability (fraction to replace)
    double beta = 1.5;     // Levy exponent (1 < beta <= 2)

    // initialize nests randomly in range
    struct Nest { double x,y,fit; };
    vector<Nest> nests_v(nests);
    for (int i = 0; i < nests; ++i) {
        nests_v[i].x = xmin + rand01() * (xmax - xmin);
        nests_v[i].y = xmin + rand01() * (xmax - xmin);
        nests_v[i].fit = fitness(nests_v[i].x, nests_v[i].y);
    }

    // track best
    double best_fit = 1e300; int best_idx = 0;
    for (int i = 0; i < nests; ++i) {
        if (nests_v[i].fit < best_fit) { best_fit = nests_v[i].fit; best_idx = i; }
    }

    // main loop
    for (int it = 0; it < iterations; ++it) {
        // 1) For each nest, generate a new solution by Levy flight and possibly replace a random nest
        for (int i = 0; i < nests; ++i) {
            // generate a 2D Levy step (independent components)
            double step_x = levy_step(beta);
            double step_y = levy_step(beta);

            double new_x = nests_v[i].x + alpha * step_x;
            double new_y = nests_v[i].y + alpha * step_y;

            // clamp to bounds
            new_x = clamp(new_x, xmin, xmax);
            new_y = clamp(new_y, xmin, xmax);

            double new_fit = fitness(new_x, new_y);

            // choose a random nest j to compare with (j != i)
            int j;
            do { j = rand() % nests; } while (j == i && nests > 1);

            // if new solution is better than nest j, replace j
            if (new_fit < nests_v[j].fit) {
                nests_v[j].x = new_x;
                nests_v[j].y = new_y;
                nests_v[j].fit = new_fit;
            }
        }

        // 2) Abandon a fraction pa of worst nests and replace them with new random ones
        int num_replace = max(1, (int)floor(pa * nests));
        // sort indices by fitness descending (worst first)
        vector<int> idx(nests);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b){ return nests_v[a].fit > nests_v[b].fit; });

        for (int r = 0; r < num_replace; ++r) {
            int worst = idx[r];
            nests_v[worst].x = xmin + rand01() * (xmax - xmin);
            nests_v[worst].y = xmin + rand01() * (xmax - xmin);
            nests_v[worst].fit = fitness(nests_v[worst].x, nests_v[worst].y);
        }

        // 3) update best
        for (int i = 0; i < nests; ++i) {
            if (nests_v[i].fit < best_fit) {
                best_fit = nests_v[i].fit;
                best_idx = i;
            }
        }

        // optional progress print
        if ((it+1) % max(1, iterations/10) == 0 || it==0) {
            cout << "Iter " << (it+1) << " best = " << best_fit << "\n";
        }
    }

    cout << "\nBest solution found: x = " << nests_v[best_idx].x
         << ", y = " << nests_v[best_idx].y << "\n";
    cout << "Best fitness = " << best_fit << "\n";

    return 0;
}

/*
------------------ Formulas used in this code ------------------

1) Lévy flight (Mantegna's method) for a scalar step:
   u ~ N(0, sigma_u^2),   v ~ N(0,1)
   step = u / |v|^(1/beta)

   where
   sigma_u = [ Gamma(1+beta) * sin(pi*beta/2) / ( Gamma((1+beta)/2) * beta * 2^((beta-1)/2) ) ]^(1/beta)

   In the code:
   step_x = levy_step(beta)
   step_y = levy_step(beta)

2) Generate new candidate by Lévy flight (component-wise):
   x_new = x_old + alpha * step_x
   y_new = y_old + alpha * step_y

3) Replacement rule (compare-and-replace):
   Choose random j (j != i).
   If f(x_new, y_new) < f(x_j, y_j) then replace:
     (x_j, y_j) := (x_new, y_new)

4) Abandonment (replace worst fraction p_a):
   Identify worst p_a * n nests and replace each with a new random point in [xmin, xmax]:
     x_new ~ U(xmin, xmax),  y_new ~ U(xmin, xmax)

5) Fitness used here (replaceable):
   f(x,y) = x^2 + y^2   (we minimize this)

Notes:
- alpha is step size scale (controls magnitude of Lévy jumps).
- beta (1 < beta <= 2) controls Lévy tail heaviness (beta=1.5 common).
- tgamma is Gamma function used in sigma_u formula.
*/