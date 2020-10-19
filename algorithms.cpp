#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include "hash.h"
#include "dataset.hpp"
#include "manhattan.h"

using namespace std;

int FindW(int samples, Dataset *set){
    int sum=0, min, tmp;
    int d = set->dimension(), img = set->getNumberOfImages();
    for(int i=0; i<samples; i++){
        min=0;
        cout << i << "/" << samples << endl;
        for(int j=0; j<img; j++){
            if(i!=j){
                if(min!=0){
                    tmp = manhattan(set->imageAt(i), set->imageAt(j), d);
                    if(tmp<min) min = tmp;
                }
                else min = manhattan(set->imageAt(i), set->imageAt(j), d);
            }
        }
        sum+=min;
    }
    return sum/samples;
}

void ANNsearch(unsigned char *image){
    // Approximate kNN
    // Input: query q
    // Let b ←Null; db ← ∞; initialize k best candidates and distances;
    // for i from 1 to L do
    //     for each item p in bucket gi(q) do
    //         if dist(q, p) < db = k-th best distance then b ← p; db ← dist(q, p)
    //         end if
    //         if large number of retrieved items (e.g. > 10L) then return b // optional
    //         end if
    //     end for
    //     return b; k best candidates;
    // end for

}

void RNGsearch(int R,int L, Dataset* query, HashTable** hashTables){
    // Approximate (r, c) Range search
    // Input: r, query q
    //     for i from 1 to L do
    //         for each item p in bucket gi(q) do
    //             if dist(q, p) < r then output p
    //             end if
    //             if large number of retrieved items (e.g. > 20L) then return // optional
    //             end if
    //         end for
    //     end for
    // return
    unsigned char* q = query->imageAt(0);
    int bucket = 0;
    for(int i= 0; i < L;i++){
        bucket = hashTables[i] ->ghash(q);
        vector<unsigned char *>* images = hashTables[i] ->getBucketArray()[bucket] -> getImageList();
        for (vector<unsigned char *>::iterator it = images->begin() ; it != images->end(); ++it){
            cout << *it << endl;
            if(manhattan(*it, q, hashTables[i] ->getvectorsDim()) < R){
                cout << "Found one item" << endl;
            }
        }
    }
}
