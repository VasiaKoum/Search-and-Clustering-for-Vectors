#include <stdlib.h>
#include <vector>
#include <iostream>
#include <cfloat>
#include <math.h>
#include <fstream>
#include <time.h>
#include <algorithm>
#include "dataset.hpp"
#include "metrics.h"
#include "lshAlgorithms.h"
#include "cubeAlgorithms.h"
#include "projection.hpp"
#include "hash.h"

void hyperTrueDistance(vector<double>& trueDist, int R, int indexq, unsigned char *q, Dataset *trainSet,  Projection* projection){
    double min, manh=0.0;
    
    if(R > 0){
        for(int i=0; i<trainSet->getNumberOfImages(); i++){
            manh = manhattan(q, trainSet->imageAt(i),  projection ->getvectorsDim());
            if(manh<R) {
                trueDist.push_back(manh);
            }
        }    
    }else{
        min = DBL_MAX;
        for(int i=0; i<trainSet->getNumberOfImages(); i++){
            manh = manhattan(q, trainSet->imageAt(i),  projection ->getvectorsDim());
            if(manh<min) {
                min = manh;
                trueDist.push_back(manh);
            }
        }
    }    

    return;

}
void hyperCubeRNGsearch(vector<Neighbor>& neighbors, int bucket, double R, int indexq, unsigned char *q, Dataset *trainSet, Projection* projection){
    unsigned int g_hash = 0;
    double manh=0.0;
    g_hash = bucket; /////////////can with loop////

    vector<imageInfo>* images = projection ->getBucketArray()[g_hash%(projection ->gethashTableSize())] -> getImageList();
    for (vector<imageInfo>::iterator it = images->begin() ; it != images->end(); ++it){
        //cout << "exw photos " << endl;
        if((*it).ghash == g_hash){
            manh = manhattan((*it).image, q, projection ->getvectorsDim());
            if(manh < R){
                neighbors.push_back(Neighbor((*it).index, manh, (*it).image));
            }
            // if(j>15*L) break;
            // j++;
        }
    }
    return;
}    

void hyperCubeANNsearch(vector<Neighbor>& neighbors, int bucket, int k, int indexq, unsigned char *q, Dataset *trainSet, Projection* projection){
    unsigned int g_hash = 0;
    double min, manh=0.0;

    
    g_hash = bucket; /////////////can with loop////
    min = DBL_MAX;
    vector<imageInfo>* images = projection ->getBucketArray()[g_hash%(projection ->gethashTableSize())] -> getImageList();
    for (vector<imageInfo>::iterator it = images->begin() ; it != images->end(); ++it){
        //cout << "exw photos " << endl;
        if((*it).ghash == g_hash){
            manh = manhattan((*it).image, q, projection ->getvectorsDim());
            if(manh < min){
                min = manh;
                neighbors.push_back(Neighbor((*it).index, min, (*it).image));
            }
            // if(j>15*L) break;
            // j++;
        }
    }
    return;
}

void hammingCombinations(int num, vector<unsigned int> & combs){
    // if(distRemain == 0){return;}
    
    int numLength = (int)log2(num)+1;

    int mask = 0;
    unsigned int newComb = 0;
    int bit = 0;
    for(int i = 0; i < numLength; i++){
        mask = 1 << i;
        bit = mask & num;
        if(bit == 0){
            bit = 1;
        }else{
            bit = 0;
        }
        newComb = (num & ~mask) | ((bit << i) & mask);
        //(combs -> at(5)).push_back(newComb);
        combs.push_back(newComb);
        // }else{
        //     hammingCombinations(newComb, combs,distRemain - 1);   
    }
    return;
}

void cubeOutput(ofstream& outputf, int indexq, int N, vector<Neighbor>& ANNneighbors, vector<Neighbor>& RNGneighbors, vector<double>& ANNtrueDist, vector<double>& RNGtrueDist,double cubeAnnTime, double cubeRngTime, double trueAnnTime, double trueRngTime){

    int j = ANNtrueDist.size()-1;
    int printn = 1;
    outputf << "Query: " << indexq << endl;
    int size = ANNneighbors.size();
    if(size > size-1-N){
        
        for(int n=size-1; n>size-1-N; n--){
            if(j>=0) ANNneighbors[n].printCubeNeighbor(printn, ANNtrueDist[j], outputf);
            j--;
            printn++;
        }
        outputf << "tHypercube: " << cubeAnnTime << endl;
        outputf << "tTrue: " << trueAnnTime << endl<< endl;
    }
    size = RNGtrueDist.size();
    j = size-1;
    printn = 1;
    outputf << "R-near neighbors:" <<endl;
    if(size > 0){
        for(int n=size-1; n>= 0; n--){
            if(j>=0) RNGneighbors[n].printCubeNeighbor(printn, RNGtrueDist[j], outputf);
            j--;
            printn++;
        }
    } 
    return;
}
    
bool compare( Neighbor& a, Neighbor &b)
{
    return a.getDist() < b.getDist();
}

void hyperCubeSearch(vector<Neighbor>& ANNneighbors,vector<Neighbor>& RNGneighbors,vector<double>& ANNtrueDist, vector<double>& RNGtrueDist, double& cubeAnnTime, double& cubeRngTime, double& trueAnnTime, double& trueRngTime,bool trueDist,int R,int N, int probs,int indexq, unsigned char* q, Dataset *trainSet, Projection * projection){
    
    unsigned int qhash = projection->ghash(q);
    int qLength = (int)log2(qhash)+1;
    // double cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime;
    clock_t cubeAnnStart, cubeRngStart, AnnTrueStart, RngTrueStart;
    vector<vector<unsigned int>> combs(probs); ///////////2d hamming combinatios for every distance

    //cout << "distance num :" << combs.size() << endl;
    bool stop = false;
    int i = 0; ///////distance 
    trueAnnTime = 0;
    trueRngTime = 0;
    cubeAnnTime = 0;
    cubeRngTime = 0;

    while(i < probs && !stop){
        
        if(i == 0){
            //cout << "first in my bucket qhash :" << qhash << endl;
            cubeAnnStart = clock();
            if(N != 0){
                hyperCubeANNsearch(ANNneighbors,qhash, projection -> getnumberOfHashFuncs(), indexq, q, trainSet, projection);
            }    
            cubeAnnTime = cubeAnnTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;

            cubeRngStart = clock();
            if(R != 0){
                hyperCubeRNGsearch(RNGneighbors, qhash, R, indexq, q,trainSet,projection);
            }
            cubeRngTime = cubeRngTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;

            if(trueDist){
                AnnTrueStart = clock();
                hyperTrueDistance(ANNtrueDist,0,indexq, q, trainSet, projection);
                trueAnnTime = trueAnnTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;

                RngTrueStart = clock();
                hyperTrueDistance(RNGtrueDist,R,indexq, q, trainSet, projection);
                trueRngTime = trueRngTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;
            }    

            // output(indexq, N, ANNneighbors, RNGneighbors, ANNtrueDist, RNGtrueDist);

        }else{
            for (int z = 0; z < combs[i-1].size(); z++){
                //search
                // vector<Neighbor> ANNneighbors;
                // vector<Neighbor> RNGneighbors;
                // vector<double> ANNtrueDist;
                // vector<double> RNGtrueDist;
                cubeAnnStart = clock();
                if(N != 0){
                    hyperCubeANNsearch(ANNneighbors,combs[i-1][z], projection -> getnumberOfHashFuncs(), indexq, q, trainSet, projection);
                }
                cubeAnnTime = cubeAnnTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;

                cubeRngStart = clock();
                if(R != 0){
                    hyperCubeRNGsearch(RNGneighbors, combs[i-1][z], R, indexq, q,trainSet,projection);
                }
                cubeRngTime = cubeRngTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;

                if(trueDist){
                    AnnTrueStart = clock();
                    hyperTrueDistance(ANNtrueDist,0,indexq, q, trainSet, projection);
                    trueAnnTime = trueAnnTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;
                    
                    RngTrueStart = clock();
                    hyperTrueDistance(RNGtrueDist,R,indexq, q, trainSet, projection);
                    trueRngTime = trueRngTime + (double)(clock() - cubeAnnStart)/CLOCKS_PER_SEC;
                }
                // output(indexq, N, ANNneighbors, RNGneighbors, ANNtrueDist, RNGtrueDist);

            }
        }
        //////////////hamming Distance////////////////////////
        if(i == 0){
            hammingCombinations(qhash, combs[i]);
        }else{
            for (int j = 0; j < combs[i-1].size(); j++){
                hammingCombinations(combs[i-1][j], combs[i]);
            }   
        }
        // if(ANNneighbors.size() > 20){
        //     break;
        // }
        
        i++; //raise distance
    }
    if(N != 0){ 
        sort(ANNneighbors.begin(), ANNneighbors.end(), compare);
        if(ANNneighbors.size() > N){
            ANNneighbors.resize(N, Neighbor(0, 0, 0));
        }    
    } 

    if(trueDist){ 
        sort(ANNtrueDist.begin(), ANNtrueDist.end());
        if(ANNtrueDist.size() > N){
            ANNtrueDist.resize(N);
        }    
    } 
}