#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

float randomRange(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

float fitness(float x, float y) {
    return sqrt(x*x + y*y); 
}

int main() {
    int n, itr;
    float minR, maxR;

    cout << "Enter number of particles: ";
    cin >> n;
    cout << "Enter number of iterations: ";
    cin >> itr;
    cout << "Enter range min and max: ";
    cin >> minR >> maxR;

    float w = 0.5, c1 = 1.5, c2 = 1.5;

    vector<float> x(n), y(n), vx(n), vy(n);
    vector<float> pbestX(n), pbestY(n);
    float gbestX, gbestY;

    for (int i = 0; i < n; i++) {
        x[i] = randomRange(minR, maxR);
        y[i] = randomRange(minR, maxR);

        vx[i] = 0;
        vy[i] = 0;

        pbestX[i] = x[i];
        pbestY[i] = y[i];
    }

    int bestIndex = 0;
    for (int i = 1; i < n; i++) {
        if (fitness(pbestX[i], pbestY[i]) < fitness(pbestX[bestIndex], pbestY[bestIndex])) {
            bestIndex = i;
        }
    }
    gbestX = pbestX[bestIndex];
    gbestY = pbestY[bestIndex];

    for (int t = 0; t < itr; t++) {
        for (int i = 0; i < n; i++) {

            float r1 = randomRange(0, 1);
            float r2 = randomRange(0, 1);

            vx[i] = w*vx[i] + c1*r1*(pbestX[i] - x[i]) + c2*r2*(gbestX - x[i]);
            vy[i] = w*vy[i] + c1*r1*(pbestY[i] - y[i]) + c2*r2*(gbestY - y[i]);

            x[i] += vx[i];
            y[i] += vy[i];

            x[i] = clamp(x[i], minR, maxR);
            y[i] = clamp(y[i], minR, maxR);

            if (fitness(x[i], y[i]) < fitness(pbestX[i], pbestY[i])) {
                pbestX[i] = x[i];
                pbestY[i] = y[i];
            }

            if (fitness(pbestX[i], pbestY[i]) < fitness(gbestX, gbestY)) {
                gbestX = pbestX[i];
                gbestY = pbestY[i];
            }
        }
    }

    cout << "\nBest Position Found: (" << gbestX << ", " << gbestY << ")";
    cout << "\nBest Distance from (0,0): " << fitness(gbestX, gbestY) << endl;

    return 0;
}
