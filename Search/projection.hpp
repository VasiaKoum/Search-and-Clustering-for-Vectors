#ifndef PROJ
#define PROJ

#include <map>
#include "hash.h"
using namespace std;

//class HashTable;

class Projection : public HashTable {
    // int bucketDim;
    // int vectorsDim;
    // HashFunction **hashFunctions;
    map<int, int>* hStorage;
    public:
        Projection(int, int, int, int, HashFunction**);
        ~Projection();
        unsigned int ghash(unsigned char*);
    
};

#endif