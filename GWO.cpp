#include <bits/stdc++.h>
using namespace std;

struct Wolf {
    double x, y;
    double fitness;
};

double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x2-x1,2) + pow(y2-y1,2));
}

int main() {
    int n, itr;
    double preyX, preyY;
    
    cout<<"Enter number of wolves: ";
    cin>>n;
    cout<<"Enter number of iterations: ";
    cin>>itr;
    cout<<"Enter prey position (x y): ";
    cin>>preyX>>preyY;

    vector<Wolf> wolves(n);

    // random initial positions
    for(int i=0; i<n; i++) {
        wolves[i].x = rand()%20; // random 0 to 20
        wolves[i].y = rand()%20;
    }

    for(int t=0; t<itr; t++) {
        
        // fitness update
        for(int i=0; i<n; i++) {
            wolves[i].fitness = distance(wolves[i].x, wolves[i].y, preyX, preyY);
        }

        // sort wolves by fitness (smallest first)
        sort(wolves.begin(), wolves.end(), [](Wolf a, Wolf b){
            return a.fitness < b.fitness;
        });

        // leaders
        Wolf alpha = wolves[0];
        Wolf beta  = wolves[1];
        Wolf delta = wolves[2];

        double a = 2 - (2.0 * t / itr);

        for(int i=0; i<n; i++) {
            double r1 = (double)rand()/RAND_MAX;
            double r2 = (double)rand()/RAND_MAX;
            double A = 2*a*r1 - a;
            double C = 2*r2;

            // update using alpha
            double Dx = abs(C*alpha.x - wolves[i].x);
            double Dy = abs(C*alpha.y - wolves[i].y);
            double X1 = alpha.x - A*Dx;
            double Y1 = alpha.y - A*Dy;

            // update using beta
            r1 = (double)rand()/RAND_MAX;
            r2 = (double)rand()/RAND_MAX;
            A = 2*a*r1 - a;
            C = 2*r2;
            Dx = abs(C*beta.x - wolves[i].x);
            Dy = abs(C*beta.y - wolves[i].y);
            double X2 = beta.x - A*Dx;
            double Y2 = beta.y - A*Dy;

            // update using delta
            r1 = (double)rand()/RAND_MAX;
            r2 = (double)rand()/RAND_MAX;
            A = 2*a*r1 - a;
            C = 2*r2;
            Dx = abs(C*delta.x - wolves[i].x);
            Dy = abs(C*delta.y - wolves[i].y);
            double X3 = delta.x - A*Dx;
            double Y3 = delta.y - A*Dy;

            // final new position (average)
            wolves[i].x = (X1 + X2 + X3)/3;
            wolves[i].y = (Y1 + Y2 + Y3)/3;
        }
    }

    cout<<"\nBest wolf found at: ("<<wolves[0].x<<", "<<wolves[0].y<<")";
    cout<<"\nDistance from prey: "<<distance(wolves[0].x, wolves[0].y, preyX, preyY);

    return 0;
}
