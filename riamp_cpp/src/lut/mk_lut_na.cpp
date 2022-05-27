#include <bitset>
#include <iostream>
#include <stdlib.h>

using namespace std;
  
int main() {
    
    bitset<8> z1(1);

    int mini = 0; 
    int maxi = 16;

    cout << "#pragma once" << endl << endl;

    cout << "alignas(32) const double na_lut["<< (maxi-mini) * 4 << "] = {" << endl;

    for (int i=mini; i<maxi; i++) {
        
        bitset<8> b(i);

        int shift = 0;
        for (int ii=0; ii<4; ii++) {

            if (       (b & z1) == 0b0) {
                cout << "    0.0";
            } else if ((b & z1) == 0b1) {
                cout << "    1.0";
            } else {
                cout << "fatal. missing something here...." << endl;
                exit(1);
            }
            
            if (i == maxi - 1 && ii == 3) {
                cout << "  // ";
            } else {
                cout << ", // ";
            }

            cout << i << " | "<< bitset<8>(i) << " >> " << shift << " = " << b << ", b & 0b1 = " << (b & z1) << ") " << endl;

            shift++;
            b >>= 1;
        }
    }

    cout << "};" << endl << endl;


    cout << "alignas(64) const int na_lut_i32["<< (maxi-mini) * 4 * 2 << "] = {" << endl;

    for (int i=mini; i<maxi; i++) {
        
        bitset<8> b(i);

        int shift = 0;
        for (int ii=0; ii<4; ii++) {

            if (       (b & z1) == 0b0) {
                cout << "    0, 0";
            } else if ((b & z1) == 0b1) {
                cout << "    0, 1";
            } else {
                cout << "fatal. missing something here...." << endl;
                exit(1);
            }
            
            if (i == maxi - 1 && ii == 3) {
                cout << "  // ";
            } else {
                cout << ", // ";
            }

            cout << i << " | "<< bitset<8>(i) << " >> " << shift << " = " << b << ", b & 0b1 = " << (b & z1) << ") " << endl;

            shift++;
            b >>= 1;
        }
    }

    cout << "};" << endl << endl;
}
