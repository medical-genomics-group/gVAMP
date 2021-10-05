#include <bitset>
#include <iostream>
#include <stdlib.h>

using namespace std;
  
int main() {
    
    bitset<8> z3(3);

    int mini = 0; 255; 
    int maxi = 256;

    cout << "#pragma once" << endl << endl;

    cout << "const double dotp_lut_a["<< (maxi-mini) * 4 << "] = {" << endl;

    for (int i=mini; i<maxi; i++) {
        
        bitset<8> b(i);        
        //cout << "b = " << b << endl;

        for (int ii=0; ii<4; ii++) {

            if (       (b & z3) == 0b00) {
                cout << "    2.0";
            } else if ((b & z3) == 0b01) {
                cout << "    0.0";
            } else if ((b & z3) == 0b10) {
                cout << "    1.0";
            } else if ((b & z3) == 0b11) {
                cout << "    0.0";
            } else {
                cout << "fatal. missing something here...." << endl;
                exit(1);
            }
            
            cout << ", // " << i << " | "<< bitset<8>(i) << " (b = " << b << ", (b & 0b11) = " << (b & z3) << ") " << endl;
            
            b >>= 2;
        }
    }

    cout << "};" << endl << endl;


    cout << "const double dotp_lut_b["<< (maxi-mini) * 4 << "] = {" << endl;

    for (int i=mini; i<maxi; i++) {
        bitset<8> b(i);

        for (int ii=0; ii<4; ii++) {

            if (       (b & z3) == 0b00) {
                cout << "    1.0";
            } else if ((b & z3) == 0b01) {
                cout << "    0.0";
            } else if ((b & z3) == 0b10) {
                cout << "    1.0";
            } else if ((b & z3) == 0b11) {
                cout << "    1.0";
            } else {
                cout << "fatal. missing something here...." << endl;
                exit(1);
            }

            cout << ", // " << i << " | "<< bitset<8>(i) << " (b = " << b << ", (b & 0b11) = " << (b & z3) << ") " << endl;

            b >>= 2;
        }
    }
    cout << "};" << endl;
}
