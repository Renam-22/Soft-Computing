#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

double triangular(double x, double a, double b, double c) {
    if (x <= a || x >= c) return 0.0;
    else if (x == b) return 1.0;
    else if (x > a && x < b) return (x - a) / (b - a);
    else return (c - x) / (c - b);
}

double trapezoidal(double x, double a, double b, double c, double d) {
    if (x <= a || x >= d) return 0.0;
    else if (x >= b && x <= c) return 1.0;
    else if (x > a && x < b) return (x - a) / (b - a);
    else return (d - x) / (d - c);
}
double singleton(double x, double a) {
    return (x == a) ? 1.0 : 0.0;
}

double sigmoid(double x, double a, double c) {
    return 1.0 / (1.0 + exp(-a * (x - c)));
}

double gaussian(double x, double mean, double sigma) {
    return exp(-0.5 * pow((x - mean) / sigma, 2));
}

int main() {
    ofstream fout("triangular_c_12.txt");
    double a = 0, b = 5, c = 12;   
    for (double x = -2; x <= 12; x += 0.5) {
        fout << x << " " << triangular(x, a, b, c) << endl;
    }
    fout.close();

    ofstream fout("trapezoidal_d_12.txt");
    double a = 0, b = 3, c = 7, d = 12; 
    for (double x = -2; x <= 12; x += 0.5) {
        fout << x << " " << trapezoidal(x, a, b, c, d) << endl;
    }
    fout.close();

     ofstream fout("singleton.txt");
    double a = 5;   
    for (double x = -2; x <= 12; x += 0.5) {
        fout << x << " " << singleton(x, a) << endl;
    }
    fout.close();

    ofstream fout("sigmoidal_c_7.txt");
    double a = 1, c = 7;   
    for (double x = -2; x <= 12; x += 0.5) {
        fout << x << " " << sigmoid(x, a, c) << endl;
    }
    fout.close();

    ofstream fout("gaussian_sigma_6.txt");
    double x = 6, mean = 2;   
    for (double sigma = -2; sigma <= 12; sigma += 0.5) {
        fout << x << " " << gaussian(x, mean, sigma) << endl;
    }
    fout.close();

    return 0;
}