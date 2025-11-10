#include <bits/stdc++.h>
using namespace std;

struct Point {
    float x, y;
};

float distanceAB(Point a, Point b) {
    return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}

int main() {
    srand(time(0));

    int n, ants, iters;
    cout << "Enter number of points: ";
    cin >> n;

    vector<Point> pts(n);
    cout << "Enter points (x y):\n";
    for(int i=0; i<n; i++) cin >> pts[i].x >> pts[i].y;

    cout << "Enter number of ants: ";
    cin >> ants;
    cout << "Enter iterations: ";
    cin >> iters;

    // ACO parameters
    float alpha = 1.0; // importance of pheromone
    float beta  = 2.0; // importance of distance
    float rho   = 0.5; // evaporation rate
    float Q     = 100; // pheromone constant

    // distance matrix
    vector<vector<float>> dist(n, vector<float>(n));
    // pheromone matrix
    vector<vector<float>> pher(n, vector<float>(n, 1.0));

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            dist[i][j] = (i==j) ? 0 : distanceAB(pts[i], pts[j]);

    for(int t=0; t<iters; t++) {

        vector<vector<int>> antPath(ants);
        vector<float> pathLen(ants, 0);

        for(int k=0; k<ants; k++) {
            int cur = 0; // fixed start point = 0
            antPath[k].push_back(cur);

            vector<int> visited(n, 0);
            visited[cur] = 1;

            while(antPath[k].size() < n) {
                vector<float> prob(n, 0);
                float sum = 0;

                for(int j=0; j<n; j++) {
                    if(!visited[j]) {
                        float p = pow(pher[cur][j], alpha) * pow(1.0/dist[cur][j], beta);
                        prob[j] = p;
                        sum += p;
                    }
                }

                for(int j=0; j<n; j++) prob[j] /= sum;

                float r = (float)rand()/RAND_MAX;
                float acc = 0;
                int next = 0;

                for(int j=0; j<n; j++) {
                    if(!visited[j]) {
                        acc += prob[j];
                        if(r <= acc) { next = j; break; }
                    }
                }

                antPath[k].push_back(next);
                visited[next] = 1;
                pathLen[k] += dist[cur][next];
                cur = next;
            }
        }

        // evaporate pheromone
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                pher[i][j] *= (1-rho);

        // add pheromone based on path quality
        for(int k=0;k<ants;k++) {
            float deposit = Q / pathLen[k];
            for(int i=0;i<n-1;i++) {
                int a = antPath[k][i];
                int b = antPath[k][i+1];
                pher[a][b] += deposit;
                pher[b][a] += deposit;
            }
        }

        cout << "Iter " << t+1 << " Best path length: " << *min_element(pathLen.begin(), pathLen.end()) << "\n";
    }

    return 0;
}