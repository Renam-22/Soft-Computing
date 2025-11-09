#include <bits/stdc++.h>
using namespace std;

// Generic fuzzy set with membership values in [0,1]
template<typename T>
class FuzzySet {
public:
    unordered_map<T,double> mu; // membership function

    FuzzySet() = default;

    // Set membership (clamped to [0,1])
    void set(const T& x, double value) {
        value = clamp(value, 0.0, 1.0);
        if (value <= 0.0) mu.erase(x);
        else mu[x] = value;
    }

    // Get membership (0 if missing)
    double get(const T& x) const {
        auto it = mu.find(x);
        return it == mu.end() ? 0.0 : it->second;
    }

    // Universe of elements present in either this or other
    static vector<T> universe(const FuzzySet<T>& A, const FuzzySet<T>& B) {
        unordered_set<T> s;
        for (auto &p : A.mu) s.insert(p.first);
        for (auto &p : B.mu) s.insert(p.first);
        return vector<T>(s.begin(), s.end());
    }

    // STANDARD OPERATIONS (return new fuzzy set)
    FuzzySet<T> complement() const {
        FuzzySet<T> R;
        for (auto &p : mu) R.set(p.first, 1.0 - p.second);
        return R;
    }

    FuzzySet<T> uni(const FuzzySet<T>& B) const { // max (standard union)
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, max(get(x), B.get(x)));
        return R;
    }

    FuzzySet<T> inter(const FuzzySet<T>& B) const { // min (standard intersection)
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, min(get(x), B.get(x)));
        return R;
    }

    // Difference A \ B = min(muA, 1 - muB)
    FuzzySet<T> difference(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, min(get(x), 1.0 - B.get(x)));
        return R;
    }

    // Algebraic sum: muA + muB - muA*muB
    FuzzySet<T> algebraicSum(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) {
            double a = get(x), b = B.get(x);
            R.set(x, a + b - a*b);
        }
        return R;
    }

    // Algebraic product: muA * muB
    FuzzySet<T> algebraicProduct(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, get(x) * B.get(x));
        return R;
    }

    // Bounded sum: min(1, muA + muB)
    FuzzySet<T> boundedSum(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, min(1.0, get(x) + B.get(x)));
        return R;
    }

    // Bounded difference: max(0, muA - muB)
    FuzzySet<T> boundedDifference(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, max(0.0, get(x) - B.get(x)));
        return R;
    }

    // Symmetric difference: |muA - muB|
    FuzzySet<T> symmetricDifference(const FuzzySet<T>& B) const {
        FuzzySet<T> R;
        auto U = universe(*this, B);
        for (auto &x : U) R.set(x, fabs(get(x) - B.get(x)));
        return R;
    }

    // Concentration: mu^p (default p=2) -- "makes set 'sharper'"
    FuzzySet<T> concentration(double p = 2.0) const {
        FuzzySet<T> R;
        if (p <= 0) return *this;
        for (auto &pr : mu) R.set(pr.first, pow(pr.second, p));
        return R;
    }

    // Dilation (power <1, e.g. sqrt)
    FuzzySet<T> dilation(double p = 0.5) const {
        FuzzySet<T> R;
        if (p <= 0) return *this;
        for (auto &pr : mu) R.set(pr.first, pow(pr.second, p));
        return R;
    }

    // Alpha-cut: crisp set of elements with mu >= alpha
    vector<T> alphaCut(double alpha) const {
        vector<T> out;
        for (auto &pr : mu) if (pr.second >= alpha) out.push_back(pr.first);
        sort(out.begin(), out.end());
        return out;
    }

    // Support: elements with mu > 0
    vector<T> support() const {
        vector<T> out;
        for (auto &pr : mu) if (pr.second > 0.0) out.push_back(pr.first);
        sort(out.begin(), out.end());
        return out;
    }

    // Core: elements with mu == 1
    vector<T> core() const {
        vector<T> out;
        for (auto &pr : mu) if (fabs(pr.second - 1.0) < 1e-12) out.push_back(pr.first);
        sort(out.begin(), out.end());
        return out;
    }

    // Height: max membership
    double height() const {
        double h = 0.0;
        for (auto &pr : mu) h = max(h, pr.second);
        return h;
    }

    // Cardinality: sigma of membership values (generalized cardinality)
    double cardinality() const {
        double s = 0.0;
        for (auto &pr : mu) s += pr.second;
        return s;
    }

    // Normalize: divide memberships by height to make height = 1 (if height>0)
    FuzzySet<T> normalize() const {
        double h = height();
        if (h <= 0.0) return *this;
        FuzzySet<T> R;
        for (auto &pr : mu) R.set(pr.first, pr.second / h);
        return R;
    }

    // Equality (approx)
    bool approxEqual(const FuzzySet<T>& B, double eps = 1e-9) const {
        auto U = universe(*this, B);
        for (auto &x : U) if (fabs(get(x) - B.get(x)) > eps) return false;
        return true;
    }

    // Pretty print
    void print(ostream& os = cout) const {
        os << "{ ";
        bool first = true;
        for (auto &pr : mu) {
            if (!first) os << ", ";
            os << pr.first << ":" << fixed << setprecision(3) << pr.second;
            first = false;
        }
        os << " }";
    }
};

// Example usage
int main() {
    FuzzySet<int> A, B;

    // construct sets (example)
    A.set(1, 0.0);
    A.set(2, 0.2);
    A.set(3, 0.6);
    A.set(4, 1.0);

    B.set(2, 0.7);
    B.set(3, 0.4);
    B.set(5, 0.9);

    cout << "A = "; A.print(); cout << "\n";
    cout << "B = "; B.print(); cout << "\n";

    auto U = A.uni(B);
    cout << "Union (max): "; U.print(); cout << "\n";

    auto I = A.inter(B);
    cout << "Intersection (min): "; I.print(); cout << "\n";

    auto C = A.complement();
    cout << "Complement of A: "; C.print(); cout << "\n";

    auto D = A.difference(B);
    cout << "A \\ B (min(muA,1-muB)): "; D.print(); cout << "\n";

    auto AS = A.algebraicSum(B);
    cout << "Algebraic sum: "; AS.print(); cout << "\n";

    auto AP = A.algebraicProduct(B);
    cout << "Algebraic product: "; AP.print(); cout << "\n";

    cout << "Alpha-cut of A (alpha=0.5): ";
    for (int x : A.alphaCut(0.5)) cout << x << " ";
    cout << "\n";

    cout << "Support of B: ";
    for (auto x : B.support()) cout << x << " ";
    cout << "\n";

    cout << "Core of A: ";
    for (auto x : A.core()) cout << x << " ";
    cout << "\n";

    cout << "Cardinality(A): " << A.cardinality() << "\n";
    cout << "Height(A): " << A.height() << "\n";

    auto NormA = A.normalize();
    cout << "Normalized A: "; NormA.print(); cout << "\n";

    auto Conc = A.concentration(2.0); // mu^2
    cout << "Concentration (A^2): "; Conc.print(); cout << "\n";

    return 0;
}
