#include <bits/stdc++.h>
using namespace std;
using Real = double;
static std::mt19937_64 RNG((uint64_t)chrono::high_resolution_clock::now().time_since_epoch().count());
static Real uniform01(){ static uniform_real_distribution<Real> d(0.0,1.0); return d(RNG); }
static Real uniformReal(Real a, Real b){ uniform_real_distribution<Real> d(a,b); return d(RNG); }
static long uniformInt(long a, long b){ uniform_int_distribution<long> d(a,b); return d(RNG); }
static Real normalReal(Real mu=0.0, Real sigma=1.0){ normal_distribution<Real> d(mu,sigma); return d(RNG); }

// ---------- utility ----------
void pause(){ cout << "Press Enter to continue..."; cin.ignore(numeric_limits<streamsize>::max(),'\n'); }

// ---------- population representations ----------
struct BinInd {
    string bits;      // bitstring
    double fitness;   // larger = better (for selection)
};
struct RealInd {
    vector<Real> x;   // real vector
    double fitness;   // larger = better
};

// default objectives (if user doesn't provide fitness):
// for binary: interpret as integer value (minimization) -> convert to fitness
long bin_to_int(const string &bits){
    long v=0;
    for(char c: bits){ v = (v<<1) + (c=='1'); }
    return v;
}
Real sphere_obj(const vector<Real>& x){
    Real s=0; for(auto &v: x) s += v*v; return s;
}

// fitness conversion for minimization: higher fitness for smaller objective
double obj_to_fitness(Real obj){
    return 1.0 / (1.0 + obj); // always positive; smaller obj => larger fitness
}

// ---------- Selection methods ----------

// Roulette wheel selection (expects fitness > 0)
int roulette_select(const vector<double>& fitness){
    double total=0;
    for(auto f: fitness) total += f;
    if(total <= 0){
        // fallback uniform
        return uniformInt(0, (long)fitness.size()-1);
    }
    Real r = uniformReal(0, total);
    Real acc=0;
    for(size_t i=0;i<fitness.size();++i){
        acc += fitness[i];
        if(r <= acc) return (int)i;
    }
    return (int)fitness.size()-1;
}

// Tournament selection (k)
int tournament_select(const vector<double>& fitness, int k){
    int best = uniformInt(0, (long)fitness.size()-1);
    for(int i=1;i<k;++i){
        int cand = uniformInt(0, (long)fitness.size()-1);
        if(fitness[cand] > fitness[best]) best = cand;
    }
    return best;
}

// Rank selection: assign probabilities proportional to rank (1..N)
int rank_select(const vector<double>& fitness){
    int N = (int)fitness.size();
    // compute ranks by sorting indexes by fitness ascending and map to ranks
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b){ return fitness[a] < fitness[b]; });
    // rank_value[i] = rank starting at 1 for worst to N for best
    vector<double> rank_prob(N);
    for(int r=0;r<N;++r){
        int id = idx[r];
        rank_prob[id] = r+1; // worse has smaller r+1
    }
    // convert to probabilities
    double sum = 0; for(int i=0;i<N;++i) sum += rank_prob[i];
    double r = uniformReal(0, sum), acc=0;
    for(int i=0;i<N;++i){
        acc += rank_prob[i];
        if(r <= acc) return i;
    }
    return N-1;
}

// Stochastic Universal Sampling (SUS): selects one index based on equally spaced pointers
int sus_select(const vector<double>& fitness){
    int N = (int)fitness.size();
    double total = 0; for(auto f: fitness) total += f;
    if(total <= 0) return uniformInt(0, N-1);
    double start = uniformReal(0, total / N);
    double pointer = start;
    double acc = 0;
    int i=0;
    for(int p=0;p<N; ++p){
        pointer = start + p * (total / N);
        while(pointer > acc && i < N){
            acc += fitness[i++];
        }
        if(i>0) return i-1;
    }
    return N-1;
}

// ---------- Crossover methods ----------

// Binary single-point
pair<string,string> single_point_crossover(const string& a, const string& b){
    int L = (int)a.size();
    if(L==0) return {a,b};
    int cp = uniformInt(1, L-1); // avoid trivial
    string c1 = a.substr(0,cp) + b.substr(cp);
    string c2 = b.substr(0,cp) + a.substr(cp);
    return {c1,c2};
}

// Binary two-point
pair<string,string> two_point_crossover(const string& a, const string& b){
    int L=(int)a.size();
    if(L<2) return {a,b};
    int p1 = uniformInt(1, L-2);
    int p2 = uniformInt(p1+1, L-1);
    string c1 = a.substr(0,p1) + b.substr(p1,p2-p1) + a.substr(p2);
    string c2 = b.substr(0,p1) + a.substr(p1,p2-p1) + b.substr(p2);
    return {c1,c2};
}

// Binary uniform crossover
pair<string,string> uniform_crossover(const string& a, const string& b){
    int L=(int)a.size();
    string c1=a, c2=b;
    for(int i=0;i<L;++i){
        if(uniform01() < 0.5){
            c1[i]=b[i];
            c2[i]=a[i];
        }
    }
    return {c1,c2};
}

// Real arithmetic per-gene crossover (alpha ~ U(0,1))
pair<vector<Real>,vector<Real>> arithmetic_crossover(const vector<Real>& p1, const vector<Real>& p2){
    int D=(int)p1.size();
    vector<Real> c1(D), c2(D);
    for(int j=0;j<D;++j){
        Real alpha = uniform01();
        c1[j] = alpha * p1[j] + (1 - alpha) * p2[j];
        c2[j] = alpha * p2[j] + (1 - alpha) * p1[j];
    }
    return {c1,c2};
}

// ---------- Mutation methods ----------

// Binary bit-flip
void bitflip_mutation(string &s, Real pm){
    for(size_t i=0;i<s.size();++i) if(uniform01() < pm) s[i] = (s[i]=='1'?'0':'1');
}

// Binary swap mutation (useful for permutations)
void swap_mutation(string &s, Real pm){
    int L=(int)s.size();
    for(int i=0;i<L;++i){
        if(uniform01() < pm){
            int j = uniformInt(0, L-1);
            swap(s[i], s[j]);
        }
    }
}

// Binary inversion mutation (reverse a segment)
void inversion_mutation(string &s, Real pm){
    if(uniform01() < pm){
        int L=(int)s.size();
        int i = uniformInt(0, L-2);
        int j = uniformInt(i+1, L-1);
        reverse(s.begin()+i, s.begin()+j+1);
    }
}

// Real Gaussian mutation
void gaussian_mutation(vector<Real> &x, Real pm, Real sigma, Real lo, Real hi){
    for(size_t j=0;j<x.size();++j){
        if(uniform01() < pm){
            x[j] += normalReal(0.0, sigma);
            if(x[j] < lo) x[j] = lo;
            if(x[j] > hi) x[j] = hi;
        }
    }
}

// Real random-reset mutation
void random_reset_mutation(vector<Real> &x, Real pm, Real lo, Real hi){
    for(size_t j=0;j<x.size();++j){
        if(uniform01() < pm){
            x[j] = uniformReal(lo, hi);
        }
    }
}

// ---------- helpers for printing ----------
void print_binpop(const vector<BinInd>& pop){
    cout << "Index | Bits       | Fitness\n";
    cout << "-----------------------------\n";
    for(size_t i=0;i<pop.size();++i){
        cout << setw(5) << i << " | " << setw(10) << pop[i].bits << " | " << pop[i].fitness << "\n";
    }
}
void print_realpop(const vector<RealInd>& pop){
    cout << "Index | Vector (x)               | Fitness\n";
    cout << "-----------------------------------------------\n";
    for(size_t i=0;i<pop.size();++i){
        cout << setw(5) << i << " | [";
        for(size_t j=0;j<pop[i].x.size();++j){
            cout << fixed << setprecision(3) << pop[i].x[j] << (j+1<pop[i].x.size()? ", ":"");
        }
        cout << "] | " << pop[i].fitness << "\n";
    }
}

// ---------- main interactive demo ----------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << "=== GA Operators Demonstration (Selection, Crossover, Mutation) ===\n\n";
    cout << "Choose representation (1 = binary, 2 = real-valued): ";
    int rep=1; cin >> rep;
    if(rep==1){
        // ---------- BINARY ----------
        int N; cout << "Enter population size N: "; cin >> N;
        int L; cout << "Bit-length per individual: "; cin >> L;
        vector<BinInd> pop(N);
        cout << "Enter each individual's bitstring (length " << L << ") optionally followed by fitness (or - to auto-calc):\n";
        for(int i=0;i<N;++i){
            cout << "Individual " << i << ": ";
            string bits; double fit;
            // read full line including possible fitness
            // consume newline
            cin >> bits;
            if((int)bits.size() != L){ cerr<<"Invalid bitstring length; abort.\n"; return 1; }
            // peek next char
            if(cin.peek()=='\n' || cin.peek()==' '){
                // try to read fitness if provided
                if(cin.peek()==' '){
                    if(!(cin >> fit)) { fit = NAN; }
                } else fit = NAN;
            } else fit = NAN;
            pop[i].bits = bits;
            if(!isnan(fit)) pop[i].fitness = fit;
            else {
                // compute default objective = integer value (minimization) -> convert by obj_to_fitness
                long val = bin_to_int(bits);
                pop[i].fitness = obj_to_fitness((Real)val);
            }
        }

        cout << "\nInitial population:\n";
        print_binpop(pop);

        // Selection method
        cout << "\nSelect selection method:\n1=Roulette 2=Tournament 3=Rank 4=SUS\nEnter choice: ";
        int selm; cin >> selm;
        int tour_k=2;
        if(selm==2){ cout << "Tournament k (e.g., 2 or 3): "; cin >> tour_k; }

        // Crossover method
        cout << "\nCrossover methods (binary): 1=single-point 2=two-point 3=uniform\nEnter choice: ";
        int xorm; cin >> xorm;
        Real crossover_prob; cout << "Crossover probability (0..1): "; cin >> crossover_prob;

        // Mutation
        cout << "\nMutation methods (binary): 1=bit-flip 2=swap 3=inversion\nEnter choice: ";
        int mutm; cin >> mutm;
        Real mut_prob; cout << "Mutation probability (per bit or per op) (0..1): "; cin >> mut_prob;

        // Perform selection to choose two parents
        // build fitness array
        vector<double> fitness(N); for(int i=0;i<N;++i) fitness[i]=pop[i].fitness;
        auto pick_index = [&](int method)->int{
            if(method==1) return roulette_select(fitness);
            if(method==2) return tournament_select(fitness, tour_k);
            if(method==3) return rank_select(fitness);
            if(method==4) return sus_select(fitness);
            return roulette_select(fitness);
        };

        cout << "\nSelecting two parents (based on chosen selection method)...\n";
        int p1 = pick_index(selm);
        int p2 = pick_index(selm);
        while(p2==p1) p2 = pick_index(selm); // ensure two different parents for demo
        cout << "Parent indices: " << p1 << " and " << p2 << "\n";
        cout << "Parent 1 bits: " << pop[p1].bits << "  fitness=" << pop[p1].fitness << "\n";
        cout << "Parent 2 bits: " << pop[p2].bits << "  fitness=" << pop[p2].fitness << "\n";

        // Crossover
        string c1 = pop[p1].bits, c2 = pop[p2].bits;
        if(uniform01() < crossover_prob){
            if(xorm==1) tie(c1,c2) = single_point_crossover(pop[p1].bits, pop[p2].bits);
            else if(xorm==2) tie(c1,c2) = two_point_crossover(pop[p1].bits, pop[p2].bits);
            else tie(c1,c2) = uniform_crossover(pop[p1].bits, pop[p2].bits);
            cout << "\nCrossover performed -> Child bits:\nChild1: " << c1 << "\nChild2: " << c2 << "\n";
        } else {
            cout << "\nCrossover not performed (probability)\n";
        }

        // Mutation
        cout << "\nApplying mutation to children...\n";
        if(mutm==1) { bitflip_mutation(c1, mut_prob); bitflip_mutation(c2, mut_prob); }
        else if(mutm==2) { swap_mutation(c1, mut_prob); swap_mutation(c2, mut_prob); }
        else { inversion_mutation(c1, mut_prob); inversion_mutation(c2, mut_prob); }

        cout << "After mutation:\nChild1: " << c1 << "\nChild2: " << c2 << "\n";

        // Show also computed fitness of children using default rule
        cout << "\nComputed fitness of children (using integer->fitness conversion):\n";
        cout << "Child1 int = " << bin_to_int(c1) << " -> fitness = " << obj_to_fitness((Real)bin_to_int(c1)) << "\n";
        cout << "Child2 int = " << bin_to_int(c2) << " -> fitness = " << obj_to_fitness((Real)bin_to_int(c2)) << "\n";

        cout << "\nDemo complete.\n";
    } else {
        // ---------- REAL ----------
        int N; cout << "Enter population size N: "; cin >> N;
        int D; cout << "Dimension (number of real variables): "; cin >> D;
        Real lo, hi; cout << "Lower bound and upper bound (space-separated): "; cin >> lo >> hi;
        vector<RealInd> pop(N);
        cout << "Enter each individual as D real numbers optionally followed by fitness (or - to auto-calc):\n";
        for(int i=0;i<N;++i){
            cout << "Individual " << i << ": ";
            vector<Real> v(D);
            for(int d=0; d<D; ++d) cin >> v[d];
            // attempt to read fitness (peek)
            double fit; if(cin.peek()==' '){ if(!(cin >> fit)) fit = NAN; } else fit = NAN;
            pop[i].x = v;
            if(!isnan(fit)) pop[i].fitness = fit;
            else {
                Real obj = sphere_obj(v);
                pop[i].fitness = obj_to_fitness(obj);
            }
        }

        cout << "\nInitial population:\n";
        print_realpop(pop);

        // Selection
        cout << "\nSelect selection method:\n1=Roulette 2=Tournament 3=Rank 4=SUS\nEnter choice: ";
        int selm; cin >> selm;
        int tour_k=2; if(selm==2){ cout << "Tournament k (e.g., 2 or 3): "; cin >> tour_k; }

        // Crossover (real): arithmetic
        cout << "\nCrossover: (only arithmetic for real) Enter crossover probability (0..1): ";
        Real crossover_prob; cin >> crossover_prob;

        // Mutation
        cout << "\nMutation methods (real): 1=gaussian 2=random-reset\nEnter choice: ";
        int mutm; cin >> mutm;
        Real mut_prob; cout << "Mutation probability (per gene) (0..1): "; cin >> mut_prob;
        Real sigma = 0.1 * (hi - lo); if(mutm==1){ cout << "Gaussian sigma (suggest 0.1*range) ["<<sigma<<"] : "; string tmp; getline(cin,tmp); }
        
        // build fitness vector
        vector<double> fitness(N); for(int i=0;i<N;++i) fitness[i]=pop[i].fitness;
        auto pick_index = [&](int method)->int{
            if(method==1) return roulette_select(fitness);
            if(method==2) return tournament_select(fitness, tour_k);
            if(method==3) return rank_select(fitness);
            if(method==4) return sus_select(fitness);
            return roulette_select(fitness);
        };

        cout << "\nSelecting two parents (based on chosen selection method)...\n";
        int p1 = pick_index(selm);
        int p2 = pick_index(selm);
        while(p2==p1) p2 = pick_index(selm);
        cout << "Parent indices: " << p1 << " and " << p2 << "\n";
        cout << "Parent1: ["; for(int j=0;j<D;++j) cout<<pop[p1].x[j]<<(j+1<D?", ":""); cout<<"] fit="<<pop[p1].fitness<<"\n";
        cout << "Parent2: ["; for(int j=0;j<D;++j) cout<<pop[p2].x[j]<<(j+1<D?", ":""); cout<<"] fit="<<pop[p2].fitness<<"\n";

        // Crossover
        vector<Real> c1 = pop[p1].x, c2 = pop[p2].x;
        if(uniform01() < crossover_prob){
            tie(c1,c2) = arithmetic_crossover(pop[p1].x, pop[p2].x);
            cout << "\nCrossover performed -> Child vectors:\nChild1: ["; for(int j=0;j<D;++j) cout<<c1[j]<<(j+1<D?", ":""); cout<<"]\n";
            cout << "Child2: ["; for(int j=0;j<D;++j) cout<<c2[j]<<(j+1<D?", ":""); cout<<"]\n";
        } else cout << "\nCrossover not performed (probability)\n";

        // Mutation
        cout << "\nApplying mutation to children...\n";
        if(mutm==1) gaussian_mutation(c1, mut_prob, sigma, lo, hi), gaussian_mutation(c2, mut_prob, sigma, lo, hi);
        else random_reset_mutation(c1, mut_prob, lo, hi), random_reset_mutation(c2, mut_prob, lo, hi);

        cout << "After mutation:\nChild1: ["; for(int j=0;j<D;++j) cout<<c1[j]<<(j+1<D?", ":""); cout<<"]\n";
        cout << "Child2: ["; for(int j=0;j<D;++j) cout<<c2[j]<<(j+1<D?", ":""); cout<<"]\n";

        // compute fitness of children using sphere objective
        cout << "\nComputed fitness of children (from sphere objective -> fitness=1/(1+obj)):\n";
        Real obj1 = sphere_obj(c1), obj2 = sphere_obj(c2);
        cout << "Child1 obj=" << obj1 << " -> fitness=" << obj_to_fitness(obj1) << "\n";
        cout << "Child2 obj=" << obj2 << " -> fitness=" << obj_to_fitness(obj2) << "\n";

        cout << "\nDemo complete.\n";
    }

    return 0;
}
