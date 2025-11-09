#include <bits/stdc++.h>
using namespace std;

// ---------------------- minimal FuzzySet<T> ----------------------
template<typename T>
class FuzzySet {
public:
    unordered_map<T,double> mu; // store only mu>0

    void set(const T& x, double v) {
        v = clamp(v, 0.0, 1.0);
        if (v <= 0.0) mu.erase(x);
        else mu[x] = v;
    }
    double get(const T& x) const {
        auto it = mu.find(x);
        return it == mu.end() ? 0.0 : it->second;
    }
    vector<T> support() const {
        vector<T> out;
        out.reserve(mu.size());
        for (auto &p: mu) out.push_back(p.first);
        sort(out.begin(), out.end());
        return out;
    }
};

// ---------------------- relation helpers ----------------------
// generic hash for pair
struct pair_hash {
    template<typename A, typename B>
    size_t operator()(const pair<A,B>& p) const noexcept {
        // a simple but common combine
        return std::hash<A>()(p.first) ^ (std::hash<B>()(p.second) << 1);
    }
};

// Relation type: map from (a,b) -> membership
template<typename A, typename B>
using FuzzyRelation = unordered_map<pair<A,B>, double, pair_hash>;

// set/get for relations (auto clamp and remove zeros)
template<typename A, typename B>
void rel_set(FuzzyRelation<A,B>& R, const A& a, const B& b, double v) {
    v = clamp(v, 0.0, 1.0);
    auto key = make_pair(a,b);
    if (v <= 0.0) R.erase(key);
    else R[key] = v;
}

template<typename A, typename B>
double rel_get(const FuzzyRelation<A,B>& R, const A& a, const B& b) {
    auto it = R.find(make_pair(a,b));
    return it == R.end() ? 0.0 : it->second;
}

// build relation from two fuzzy sets (cartesian product) using a provided membership function f(a,b)
// This helper lets you create relations easily from sets (optional).
template<typename A, typename B>
FuzzyRelation<A,B> build_relation_from_sets(const FuzzySet<A>& X, const FuzzySet<B>& Y,
                                             function<double(const A&, const B&)> f) {
    FuzzyRelation<A,B> R;
    for (auto a : X.support())
        for (auto b : Y.support()) {
            double v = clamp(f(a,b), 0.0, 1.0);
            if (v > 0) R[{a,b}] = v;
        }
    return R;
}

// ---------------------- relation operations ----------------------

// Union: μ_{R∪S}(a,b) = max( μ_R(a,b), μ_S(a,b) )
template<typename A, typename B>
FuzzyRelation<A,B> rel_union(const FuzzyRelation<A,B>& R, const FuzzyRelation<A,B>& S) {
    FuzzyRelation<A,B> T = R; // copy entries from R
    for (auto &p : S) {
        auto key = p.first;
        double v = max( rel_get(R, key.first, key.second), p.second );
        if (v <= 0.0) T.erase(key); else T[key] = v;
    }
    return T;
}

// Intersection: μ_{R∩S}(a,b) = min( μ_R(a,b), μ_S(a,b) )
template<typename A, typename B>
FuzzyRelation<A,B> rel_intersection(const FuzzyRelation<A,B>& R, const FuzzyRelation<A,B>& S) {
    // iterate over union of keys to ensure all pairs are considered
    FuzzyRelation<A,B> T;
    unordered_set<size_t> seen;
    for (auto &p : R) {
        auto k = p.first;
        double v = min(p.second, rel_get(S, k.first, k.second));
        if (v > 0.0) T[k] = v;
        seen.insert( (size_t)(&p) ); // noop: we don't need this; kept simple
    }
    // also consider keys only present in S but not in R
    for (auto &p : S) {
        auto k = p.first;
        if (R.find(k) == R.end()) {
            double v = min(rel_get(R,k.first,k.second), p.second); // effectively 0
            if (v > 0.0) T[k] = v;
        }
    }
    return T;
}

// Complement: μ_{~R}(a,b) = 1 - μ_R(a,b)
// Important: complement will only include keys present in R (others implicitly 0 => complement 1 but
// storing an entry (a,b)=1 for every possible pair is usually not wanted). We will return complement only for keys present in R.
template<typename A, typename B>
FuzzyRelation<A,B> rel_complement(const FuzzyRelation<A,B>& R) {
    FuzzyRelation<A,B> T;
    for (auto &p : R) {
        T[p.first] = clamp(1.0 - p.second, 0.0, 1.0);
    }
    return T;
}

// Max-Min Composition: (R ◦ S)(a,c) = max_{b} min( R(a,b), S(b,c) )
// We auto-extract the mid-universe B from keys present in R and S.
template<typename A, typename B, typename C>
FuzzyRelation<A,C> rel_compose_maxmin(const FuzzyRelation<A,B>& R, const FuzzyRelation<B,C>& S) {
    // collect A, B, C universes
    unordered_set<A> A_univ;
    unordered_set<B> B_univ;
    unordered_set<C> C_univ;
    for (auto &p : R) { A_univ.insert(p.first.first); B_univ.insert(p.first.second); }
    for (auto &p : S) { B_univ.insert(p.first.first); C_univ.insert(p.first.second); }

    FuzzyRelation<A,C> T;
    for (auto a : A_univ) {
        for (auto c : C_univ) {
            double best = 0.0;
            for (auto b : B_univ) {
                double v = min( rel_get(R,a,b), rel_get(S,b,c) );
                if (v > best) best = v;
            }
            if (best > 0.0) T[{a,c}] = best;
        }
    }
    return T;
}

// Pretty print relation
template<typename A, typename B>
void print_relation(const FuzzyRelation<A,B>& R) {
    if (R.empty()) { cout << "{ }\n"; return; }
    cout << "{ ";
    bool first = true;
    for (auto &p : R) {
        if (!first) cout << ", ";
        cout << "(" << p.first.first << "," << p.first.second << "):" << fixed << setprecision(3) << p.second;
        first = false;
    }
    cout << " }\n";
}

// ---------------------- Example usage ----------------------
int main() {
    // Build two fuzzy sets (these could be domains X and Y)
    FuzzySet<int> X; X.set(1, 1.0); X.set(2, 1.0);
    FuzzySet<int> Y; Y.set(1, 1.0); Y.set(2, 1.0);

    // Build relations R (X->Y) and S (Y->X) directly using rel_set
    FuzzyRelation<int,int> R, S;
    rel_set(R, 1, 1, 0.5); rel_set(R, 1, 2, 0.2); rel_set(R, 2, 1, 0.9); rel_set(R, 2, 2, 0.7);
    rel_set(S, 1, 1, 0.3); rel_set(S, 1, 2, 0.8); rel_set(S, 2, 1, 0.6); rel_set(S, 2, 2, 0.4);

    cout << "R = "; print_relation(R);
    cout << "S = "; print_relation(S);

    auto U = rel_union(R,S);
    cout << "Union (R ∪ S) = "; print_relation(U);

    auto I = rel_intersection(R,S);
    cout << "Intersection (R ∩ S) = "; print_relation(I);

    auto C = rel_complement(R);
    cout << "Complement (~R) [for stored keys only] = "; print_relation(C);

    auto T = rel_compose_maxmin(R,S);
    cout << "Composition T = R ◦ S (max-min) = "; print_relation(T);

    return 0;
}
