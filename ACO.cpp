#include <bits/stdc++.h>
using namespace std;

struct Node { double x, y; };

double dist(const Node &a, const Node &b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}

double rnd01() {
    return (double)rand() / RAND_MAX;
}

int roulette(const vector<double> &w) {
    double sum = 0;
    for (double v : w) sum += v;
    if (sum == 0) return -1;
    double r = rnd01()*sum, s = 0;
    for (int i = 0; i < w.size(); i++) {
        s += w[i];
        if (r <= s) return i;
    }
    return w.size()-1;
}

int main() {
    srand(time(0));

    int n, ants, iter;
    cout<<"Nodes count: ";
    cin>>n;

    vector<Node> node(n);
    cout<<"Enter x y for nodes:\n";
    for(int i=0;i<n;i++) cin>>node[i].x>>node[i].y;

    cout<<"Ants: "; cin>>ants;
    cout<<"Iterations: "; cin>>iter;

    // parameters
    double alpha=1, beta=5, rho=0.5, Q=100, tau0=1;

    vector<vector<double>> d(n, vector<double>(n)), eta(n, vector<double>(n)), tau(n, vector<double>(n,tau0));

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            if(i==j) continue;
            d[i][j] = dist(node[i],node[j]);
            eta[i][j] = 1.0/d[i][j];
        }

    double bestLen=1e9;
    vector<int> bestTour;

    for(int it=0; it<iter; it++){
        vector<vector<int>> allTours(ants);
        vector<double> lens(ants);

        for(int k=0;k<ants;k++){
            vector<int> tour;
            vector<bool> vis(n,false);

            int cur = 0;
            tour.push_back(cur);
            vis[cur]=true;

            while(tour.size() < n){
                vector<double> w(n,0);
                for(int j=0;j<n;j++)
                    if(!vis[j])
                        w[j] = pow(tau[cur][j],alpha) * pow(eta[cur][j],beta);

                int next = roulette(w);
                if(next==-1) for(int j=0;j<n;j++) if(!vis[j]){ next=j; break; }

                vis[next]=true;
                tour.push_back(next);
                cur = next;
            }

            double L=0;
            for(int i=0;i<n-1;i++) L += d[tour[i]][tour[i+1]];
            L += d[tour.back()][tour[0]];

            allTours[k]=tour;
            lens[k]=L;

            if(L<bestLen){ bestLen=L; bestTour=tour; }
        }

        // evaporation
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                tau[i][j] *= (1-rho);

        // deposit
        for(int k=0;k<ants;k++){
            double add = Q/lens[k];
            for(int i=0;i<n-1;i++){
                int a=allTours[k][i], b=allTours[k][i+1];
                tau[a][b] += add;
                tau[b][a] += add;
            }
            int a=allTours[k].back(), b=allTours[k][0];
            tau[a][b] += add;
            tau[b][a] += add;
        }

        if(it%10==0 || it==iter-1)
            cout<<"Iter "<<it+1<<" Best="<<bestLen<<"\n";
    }

    cout<<"\nFinal Best Length: "<<bestLen<<"\nBest Path: ";
    for(int x:bestTour) cout<<x<<" ";
}