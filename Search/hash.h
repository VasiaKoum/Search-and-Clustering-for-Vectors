#include <vector>
#include <iostream>
#include <stdlib.h>

class HashFunction{
    int *s_numbers;
    int d; //d dimension
    int w; //w window
    int M; //M = 2^(32/k)
    int m; //m = 2^(32-3) ,must m > max_xi
    int k; //k number of hashFunctions

    public:
        HashFunction(int, int, int);
        ~HashFunction();
        int hash(unsigned char*);
        int getM(){ return M; }
};

struct imageInfo{
    int index;
    unsigned int ghash;
    unsigned char * image;

    public:

    imageInfo(int i, int j, unsigned char * im): index(i), ghash(j), image(im){};
};

class Bucket{

    int bucketValue;
    std::vector<imageInfo> images;

    public:
        Bucket(int);
        void addImage(int,int,unsigned char *);
        imageInfo *popBackImage();
        std::vector<imageInfo>* getImageList(){return &images;}
};

class HashTable{
    protected:
        int vectorsDim;
        int hashTableSize;      //n/16 number of buckets
        int numberOfHashFuncs; //k
        HashFunction **hashFunctions;
        Bucket **bucketArray;

    public:
        HashTable(int, int, int, int);
        ~HashTable();
        int gethashTableSize() {return hashTableSize;}
        int getvectorsDim() {return vectorsDim;}
        int getnumberOfHashFuncs() {return numberOfHashFuncs;}
        HashFunction** getHashFunctions(){return hashFunctions;}
        Bucket** getBucketArray(){return bucketArray;}

        unsigned int ghash(unsigned char *);
        // void ANNsearch(unsigned char *);
        // void RNGsearch();
};
