// ga_with_mutation.cpp
// Simple Genetic Algorithm step: create -> fitness -> select -> crossover -> mutate
// Supports Binary and (x,y) real individuals.
// Mutation:
//   - Binary: bit-flip (per-bit probability pm_bit)
//   - (x,y): Gaussian mutation (per-coordinate prob pm_xy, std = sigma), clamped to [xmin,xmax]

#include <bits/stdc++.h>
using namespace std;

// ------------------ Random helpers ------------------
double rnd01() { return (double)rand() / RAND_MAX; }
double randRange(double lo, double hi) { return lo + rnd01() * (hi - lo); }

// simple normal(0,1) using Box-Muller
double randn() {
    static bool has = false;
    static double spare;
    if (has) { has = false; return spare; }
    double u, v, s;
    do {
        u = 2.0 * rnd01() - 1.0;
        v = 2.0 * rnd01() - 1.0;
        s = u*u + v*v;
    } while (s == 0.0 || s >= 1.0);
    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul;
    has = true;
    return u * mul;
}

string randomBinaryString(int len) {
    string s; s.reserve(len);
    for (int i = 0; i < len; ++i) s.push_back((rand() % 2) ? '1' : '0');
    return s;
}

// clamp value
double clampVal(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// ------------------ Population creators ------------------
vector<string> createBinaryPopulation(int n, int len) {
    vector<string> pop; pop.reserve(n);
    for (int i = 0; i < n; ++i) pop.push_back(randomBinaryString(len));
    return pop;
}

vector<pair<double,double>> createXYPopulation(int n, double xmin, double xmax) {
    vector<pair<double,double>> pop; pop.reserve(n);
    for (int i = 0; i < n; ++i) pop.emplace_back(randRange(xmin, xmax), randRange(xmin, xmax));
    return pop;
}

// ------------------ Fitness (examples) ------------------
// Binary: minimize number of zeros (i.e. maximize ones)
vector<double> evalBinaryFitness(const vector<string> &pop) {
    vector<double> f; f.reserve(pop.size());
    for (auto &s : pop) {
        int ones = 0;
        for (char c : s) if (c == '1') ++ones;
        f.push_back((double)(s.size() - ones)); // smaller = better
    }
    return f;
}

// (x,y): sphere function
vector<double> evalXYFitness(const vector<pair<double,double>> &pop) {
    vector<double> f; f.reserve(pop.size());
    for (auto &p : pop) f.push_back(p.first*p.first + p.second*p.second);
    return f;
}

// ------------------ Selection helpers ------------------
double fitnessToScore(double f) { return 1.0 / (1.0 + f); } // for minimization
int rouletteIndex(const vector<double> &scores) {
    double sum = 0; for (double v : scores) sum += v;
    if (sum <= 0) return rand() % scores.size();
    double r = rnd01() * sum, cum = 0;
    for (int i = 0; i < (int)scores.size(); ++i) {
        cum += scores[i];
        if (r <= cum) return i;
    }
    return (int)scores.size() - 1;
}
int tournamentIndex(const vector<double> &fitness, int k=2) {
    int n = fitness.size();
    int best = rand() % n; double bestFit = fitness[best];
    for (int i = 1; i < k; ++i) {
        int cand = rand() % n;
        if (fitness[cand] < bestFit) { best = cand; bestFit = fitness[cand]; }
    }
    return best;
}
int rankIndex(const vector<double> &fitness) {
    int n = fitness.size();
    vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a,int b){ return fitness[a] < fitness[b]; }); // best first
    vector<double> weight(n,0); double sum = 0;
    for (int r = 0; r < n; ++r) { weight[idx[r]] = (double)(n - r); sum += weight[idx[r]]; }
    double r = rnd01() * sum, cum = 0;
    for (int i = 0; i < n; ++i) { cum += weight[i]; if (r <= cum) return i; }
    return n-1;
}

// wrappers to produce parent index pairs
vector<pair<int,int>> selectParentsRoulette(const vector<double> &fitness, int numPairs) {
    int n = fitness.size();
    vector<double> scores(n);
    for (int i = 0; i < n; ++i) scores[i] = fitnessToScore(fitness[i]);
    vector<pair<int,int>> out; out.reserve(numPairs);
    for (int i = 0; i < numPairs; ++i) out.emplace_back(rouletteIndex(scores), rouletteIndex(scores));
    return out;
}
vector<pair<int,int>> selectParentsTournament(const vector<double> &fitness, int numPairs, int k=2) {
    vector<pair<int,int>> out; out.reserve(numPairs);
    for (int i = 0; i < numPairs; ++i) out.emplace_back(tournamentIndex(fitness,k), tournamentIndex(fitness,k));
    return out;
}
vector<pair<int,int>> selectParentsRank(const vector<double> &fitness, int numPairs) {
    vector<pair<int,int>> out; out.reserve(numPairs);
    for (int i = 0; i < numPairs; ++i) out.emplace_back(rankIndex(fitness), rankIndex(fitness));
    return out;
}

// ------------------ Crossover ------------------
// Binary single-point
pair<string,string> crossoverBinary(const string &a, const string &b) {
    int len = a.size();
    if (len <= 1) return {a,b};
    int cut = 1 + rand() % (len - 1);
    string c1 = a.substr(0,cut) + b.substr(cut);
    string c2 = b.substr(0,cut) + a.substr(cut);
    return {c1,c2};
}
// Arithmetic (x,y)
pair<pair<double,double>,pair<double,double>> crossoverXY(const pair<double,double> &p1, const pair<double,double> &p2) {
    double alpha = rnd01();
    double x1 = alpha*p1.first + (1-alpha)*p2.first;
    double y1 = alpha*p1.second + (1-alpha)*p2.second;
    double x2 = (1-alpha)*p1.first + alpha*p2.first;
    double y2 = (1-alpha)*p1.second + alpha*p2.second;
    return {{x1,y1},{x2,y2}};
}

// ------------------ Build children from parent pairs ------------------
vector<string> buildBinaryNextGen(const vector<string> &pop, const vector<pair<int,int>> &pairs, int popSize) {
    vector<string> children; children.reserve(popSize);
    for (auto &pr : pairs) {
        auto ch = crossoverBinary(pop[pr.first], pop[pr.second]);
        children.push_back(ch.first);
        if ((int)children.size() < popSize) children.push_back(ch.second);
        if ((int)children.size() >= popSize) break;
    }
    while ((int)children.size() < popSize) children.push_back(pop[rand() % pop.size()]);
    return children;
}
vector<pair<double,double>> buildXYNextGen(const vector<pair<double,double>> &pop, const vector<pair<int,int>> &pairs, int popSize) {
    vector<pair<double,double>> children; children.reserve(popSize);
    for (auto &pr : pairs) {
        auto ch = crossoverXY(pop[pr.first], pop[pr.second]);
        children.push_back(ch.first);
        if ((int)children.size() < popSize) children.push_back(ch.second);
        if ((int)children.size() >= popSize) break;
    }
    while ((int)children.size() < popSize) children.push_back(pop[rand() % pop.size()]);
    return children;
}

// ------------------ Mutation ------------------

// Binary bit-flip mutation: pm_bit = probability to flip each bit
void mutateBinary(vector<string> &pop, double pm_bit) {
    for (auto &s : pop) {
        for (int i = 0; i < (int)s.size(); ++i) {
            if (rnd01() < pm_bit) s[i] = (s[i] == '1') ? '0' : '1';
        }
    }
}

// (x,y) Gaussian mutation:
// pm_xy = probability to mutate each coordinate (x or y)
// sigma = standard deviation for gaussian noise (set relative to domain)
void mutateXY(vector<pair<double,double>> &pop, double pm_xy, double sigma, double xmin, double xmax) {
    for (auto &p : pop) {
        if (rnd01() < pm_xy) {
            p.first += randn() * sigma;
            p.first = clampVal(p.first, xmin, xmax);
        }
        if (rnd01() < pm_xy) {
            p.second += randn() * sigma;
            p.second = clampVal(p.second, xmin, xmax);
        }
    }
}

// ------------------ MAIN (glue all) ------------------
int main() {
    srand((unsigned)time(nullptr));
    cout.setf(std::ios::fixed);
    cout << setprecision(6);

    int popSize, generations;
    cout << "Enter population size: "; cin >> popSize;
    cout << "Enter number of generations: "; cin >> generations;
    if (popSize < 2) { cout << "Population must be >= 2\n"; return 0; }

    cout << "Choose data type (1 = Binary, 2 = (x,y) real): ";
    int type; cin >> type;

    vector<string> binPop;
    vector<pair<double,double>> xyPop;
    int chromLen = 0;
    double xmin=0, xmax=0;

    if (type == 1) {
        cout << "Enter chromosome length (bits): "; cin >> chromLen;
        binPop = createBinaryPopulation(popSize, chromLen);
    }
    else {
        cout << "Enter xmin xmax: "; cin >> xmin >> xmax;
        xyPop = createXYPopulation(popSize, xmin, xmax);
    }

    double pm_bit = 0.01;
    double pm_xy = 0.2;
    double sigma_rel = 0.05;

    for (int gen = 1; gen <= generations; gen++) {
        cout << "\n========== GENERATION " << gen << " ==========\n";

        vector<double> fitness;
        if (type == 1) fitness = evalBinaryFitness(binPop);
        else fitness = evalXYFitness(xyPop);

        cout << "Choose selection method (1=Roulette, 2=Tournament, 3=Rank): ";
        int sel; cin >> sel;

        int numPairs = (popSize + 1) / 2;
        vector<pair<int,int>> parentPairs;

        if (sel == 1) parentPairs = selectParentsRoulette(fitness, numPairs);
        else if (sel == 2) {
            int k; cout << "Tournament size k: "; cin >> k;
            parentPairs = selectParentsTournament(fitness, numPairs, k);
        }
        else parentPairs = selectParentsRank(fitness, numPairs);

        // ---- Crossover ----
        if (type == 1) {
            auto children = buildBinaryNextGen(binPop, parentPairs, popSize);
            mutateBinary(children, pm_bit);

            // ✅ store back to population for next generation
            binPop = children;
        }
        else {
            auto children = buildXYNextGen(xyPop, parentPairs, popSize);
            double sigma = sigma_rel * (xmax - xmin);
            mutateXY(children, pm_xy, sigma, xmin, xmax);

            // ✅ store back to population for next generation
            xyPop = children;
        }

        // Print population after mutation
        cout << "Population after mutation:\n";
        if (type == 1) {
            for (int i = 0; i < popSize; i++)
                cout << i << ": " << binPop[i] << "\n";
        } else {
            for (int i = 0; i < popSize; i++)
                cout << i << ": (" << xyPop[i].first << ", " << xyPop[i].second << ")\n";
        }
    }

    cout << "\nGA process completed!\n";
    return 0;
}
