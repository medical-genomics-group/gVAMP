//
//  gadgets.hpp
//  gctb
//
//  Created by Jian Zeng on 14/06/2016.
//  Copyright © 2016 Jian Zeng. All rights reserved.
//

#ifndef toolbox_hpp
#define toolbox_hpp

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

namespace Gadget {

class Timer {
    time_t prev, curr;
public:
    Timer(){
        setTime();
    };
    void setTime(void);
    time_t getTime(void);
    time_t getElapse(void);
    string format(const time_t time);
    string getDate(void);
};

class Tokenizer : public vector<string> {
    // adopted from matvec
public:
    void getTokens(const string &str, const string &sep);
    int  getIndex(const string &str);
};

template <class T> class Recoder : public map<T,unsigned> {
    // adopted from matvec
    unsigned count;
public:
    Recoder(void){count=0;}
    unsigned code(T s){
        typename map<T,unsigned>::iterator mapit = this->find(s);
        if(mapit == this->end()){
            (*this)[s] = ++count;
            return count;
        }
        else {
            return (*mapit).second;
        }
    }
    void display_codes(ostream & os = cout){
        typename Recoder::iterator it;
        for (it=this->begin(); it!=this->end();it++){
            os << (*it).first << " " << (*it).second << endl;
        }
    }
};

}

#endif /* toolbox_hpp */
