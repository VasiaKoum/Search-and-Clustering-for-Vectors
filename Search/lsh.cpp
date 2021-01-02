#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "hash.hpp"
#include "dataset.hpp"
#include "lshAlgorithms.hpp"
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
// ./lsh -d train-images.idx3-ubyte -R 1.0 -q fileq -k 4 -L 5 -o fileo -N 1

using namespace std;

int main(int argc, char** argv){
    if (argc>6 && argc<16){
        char *d=NULL, *q=NULL, *o=NULL, *k=NULL, *l=NULL, *n=NULL, *r=NULL;
        double R=1.0, exec_time;
        int K=4, L=5, N=1;
        bool vars[7] = { 0 };
        for (int i=0; i<7; i++) vars[i] = true;
        for (int i = 0; i<argc; i++){
            if (!strcmp("-d", argv[i]) && vars[0]) {
                d = (char*)argv[i+1];   /* -inputfile */
                vars[0] = false;
            }
            if (!strcmp("-q", argv[i]) && vars[1]) {
                q = (char*)argv[i+1];   /* -queryfile */
                vars[1] = false;
            }
            if (!strcmp("-k", argv[i]) && vars[2]) {
                k = (char*)argv[i+1];   /* -k */
                vars[2] = false;
            }
            if (!strcmp("-L", argv[i]) && vars[3]) {
                l = (char*)argv[i+1];   /* -L */
                vars[3] = false;
            }
            if (!strcmp("-o", argv[i]) && vars[4]) {
                o = (char*)argv[i+1];   /* -o */
                vars[4] = false;
            }
            if (!strcmp("-N", argv[i]) && vars[5]) {
                n = (char*)argv[i+1];   /* -N */
                vars[5] = false;
            }
            if (!strcmp("-R", argv[i]) && vars[6]) {
                r = (char*)argv[i+1];   /* -R */
                vars[6] = false;
            }
        }

        if(d==NULL || q==NULL || o==NULL){
            cout << "You must run the program with parameters(REQUIRED): –d <input file> –q <query file>" << endl;
            cout << "With additional parameters: –k <int> -L <int> -ο <output file> -Ν <number of nearest> -R <radius>" << endl;
            return(0);
        }
        else{
            if (k!=NULL) K = atoi(k);
            if (l!=NULL) L = atoi(l);
            if (n!=NULL) N = atoi(n);
            if (r!=NULL) R = atof(r);

            unsigned int magicNumber = 0,numberOfImages = 0,numberOfRows = 0,numberOfColumns = 0, img=0;
            int numOfpixels;

            //Open train file
            ifstream trainInput;
            // if(!trainInput.is_open()){
            //     cerr<<"Failed to open input data."<<endl;
            //     return(0);
            // }
            
            trainInput.open(d, std::ios::binary);
            if(!trainInput.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return(0);
            }
            cout << "check" << endl;
            trainInput.read((char*)&magicNumber, 4);
            trainInput.read((char*)&numberOfImages, 4);
            trainInput.read((char*)&numberOfRows, 4);
            trainInput.read((char*)&numberOfColumns, 4);
            cout << numberOfImages << endl;
            //Convert intergers from Big Endian to Little Endian
            magicNumber = SWAP_INT32(magicNumber);
            numberOfImages = SWAP_INT32(numberOfImages);
            numberOfRows = SWAP_INT32(numberOfRows);
            numberOfColumns = SWAP_INT32(numberOfColumns);
            cout << numberOfImages << endl;

            Dataset trainSet(magicNumber, numberOfImages, numberOfColumns, numberOfRows);
            trainInput.read((char*)trainSet.imageAt(0), (trainSet.getNumberOfPixels())*(trainSet.getNumberOfImages()));
            trainInput.close();

            img = numberOfImages;
            string queryfile, outputfile, answer;
            bool termination = false;

            while(!termination){
                /* PROGRAM STARTS HERE */
                clock_t tStart = clock();

                //Open query file
                fstream queryInput(q);
                if(!queryInput.is_open()){
                    cerr<<"Failed to open input data."<<endl;
                    return 0;
                }
                queryInput.read((char*)&magicNumber, 4);
                queryInput.read((char*)&numberOfImages, 4);
                queryInput.read((char*)&numberOfRows, 4);
                queryInput.read((char*)&numberOfColumns, 4);

                //Convert intergers from Big Endian to Little Endian
                magicNumber = SWAP_INT32(magicNumber);
                numberOfImages = SWAP_INT32(numberOfImages);
                numberOfRows = SWAP_INT32(numberOfRows);
                numberOfColumns = SWAP_INT32(numberOfColumns);
                Dataset querySet(magicNumber, numberOfImages, numberOfColumns, numberOfRows);
                queryInput.read((char*)querySet.imageAt(0), (querySet.getNumberOfPixels())*(querySet.getNumberOfImages()));
                queryInput.close();


                ///////////////////////////////////////structure test///////////////////////////////////////
                int bucketsNumber = floor(trainSet.getNumberOfImages()/16);

                // int W = FindW(img, &trainSet);
                // cout << "W is " << W << endl;
                int W = 40000;
                HashFunction** hashFamily = new HashFunction*[trainSet.getNumberOfPixels()];
                for(int i = 0; i < trainSet.getNumberOfPixels();i++){
                    hashFamily[i] = NULL;
                }

                HashTable **hashTables = new HashTable*[L];
                for(int i=0; i<L; i++){
                    hashTables[i] = new HashTable(trainSet.getNumberOfPixels(),bucketsNumber, K,W,hashFamily);

                    for(int j=0; j<img; j++){
                        unsigned int g_hash = (unsigned int)(hashTables[i]->ghash(trainSet.imageAt(j)));
                        hashTables[i]->getBucketArray()[g_hash%bucketsNumber]->addImage(j,g_hash,trainSet.imageAt(j));
                    }
                }

                ofstream outputf(o);
                if (!outputf.is_open()){
                    cerr<<"Failed to open output data."<<endl;
                    return 0;
                }
                clock_t lshAnnStart, lshRngStart, AnnTrueStart, RngTrueStart;
                double lshAnnTime, lshRngTime, trueAnnTime, trueRngTime;
                for(int index=0; index<numberOfImages; index++){
                    vector<Neighbor> ANNneighbors;
                    vector<Neighbor> RNGneighbors;
                    vector<double> ANNtrueDist;
                    vector<double> RNGtrueDist;

                    lshAnnStart = clock();
                    ANNsearch(ANNneighbors,L, N, querySet.imageAt(index), hashTables);
                    lshAnnTime = (double)(clock() - lshAnnStart)/CLOCKS_PER_SEC;

                    AnnTrueStart = clock();
                    trueDistance(ANNtrueDist, 0, querySet.imageAt(index), &trainSet,hashTables);
                    sort(ANNtrueDist.begin(), ANNtrueDist.end());
                    trueAnnTime = (double)(clock() - AnnTrueStart)/CLOCKS_PER_SEC;

                    lshRngStart = clock();
                    RNGsearch(RNGneighbors, L, R, querySet.imageAt(index), hashTables);
                    lshRngTime = (double)(clock() - lshRngStart)/CLOCKS_PER_SEC;

                    // RngTrueStart = clock();
                    // trueDistance(RNGtrueDist, R, querySet.imageAt(index), &trainSet,hashTables);
                    // trueRngTime = (double)(clock() - RngTrueStart)/CLOCKS_PER_SEC;

                    int size = ANNneighbors.size();
                    int j = 0;
                    int printi = 1;
                    outputf << "Query: " << index << endl;
                    if(size > size-1-N){
                        for(int i=size-1; i>size-1-N; i--){
                            if(j>=0) ANNneighbors[i].printLshNeighbor(printi, ANNtrueDist[j],false, outputf);
                            j++;
                            printi++;
                        }
                        outputf << "tLSH: " << lshAnnTime << endl;
                        outputf << "tTrue: " << trueAnnTime<< endl<< endl;
                    }
                    size = RNGneighbors.size();
                    j = 0;
                    printi = 1;
                    outputf << "R-near neighbors:" <<endl;
                    if(size > 0){
                        for(int i=size-1; i>= 0; i--){
                            if(j>=0) RNGneighbors[i].printLshNeighbor(printi, 0,true, outputf);
                            j++;
                            printi++;
                        }
                    }
                }
                outputf.close();

                for(int i=0; i<trainSet.getNumberOfPixels(); i++){
                    if(hashFamily[i]!=NULL){
                        delete hashFamily[i];
                    }
                }
                delete[] hashFamily;
                for(int i=0; i<L; i++) delete hashTables[i];
                delete[] hashTables;

                /* PROGRAM ENDS HERE */
                exec_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
                cout << "\nExecution time is: "<< exec_time << endl;

                cout << "\nYou want to execute the lsh with another queryfile? (Y/N)" << endl;
                cin >> answer;
                if(answer.compare("Y")==0 || answer.compare("Yes")==0) {
                    cout << "Please type the path for queryfile:" << endl;
                    cin >> queryfile;
                    q = &queryfile[0];
                    cout << "Please type the path for outputfile:" << endl;
                    cin >> outputfile;
                    o = &outputfile[0];
                }
                else if (answer.compare("N")==0 || answer.compare("No")==0){
                    termination = true;
                    cout << "The program will terminate." << endl;
                }
                else {
                    cout << "This answer is not recognizable. The program will terminate." << endl;
                    termination = true;
                }


            }
        }

    }
    else {
        cout << "You must run the program with parameters(REQUIRED): –d <input file> –q <query file>" << endl;
        cout << "With additional parameters: –k <int> -L <int> -ο <output file> -Ν <number of nearest> -R <radius>" << endl;
    }
}
