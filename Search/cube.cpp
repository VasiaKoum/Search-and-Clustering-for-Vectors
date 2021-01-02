#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <fstream>
#include <math.h>
#include "dataset.hpp"
#include "projection.hpp"
#include "cubeAlgorithms.hpp"
#include "lshAlgorithms.hpp"
#define IMAGESIZE 800
#define SWAP_INT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))
// ./cube -d train-images.idx3-ubyte -q fileq -k 3 -M 10 -probes 2 -ο output -Ν 1 -R 1.0

using namespace std;

int main(int argc, char** argv){
    if (argc>6 && argc<18){
        char *d=NULL, *q=NULL, *o=NULL, *k=NULL, *m=NULL, *pr=NULL, *n=NULL, *r=NULL;
        double R=1.0, exec_time;
        int K=3, probes=2, N=5;
        int Max=10;
        bool vars[8] = { 0 };
        for (int i=0; i<8; i++) vars[i] = true;
        for (int i=0; i<argc; i++){
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
            if (!strcmp("-M", argv[i]) && vars[3]) {
                m = (char*)argv[i+1];   /* -M */
                vars[3] = false;
            }
            if (!strcmp("-probes", argv[i]) && vars[4]) {
                pr = (char*)argv[i+1]; /* -probes */
                vars[4] = false;
            }
            if (!strcmp("-o", argv[i]) && vars[5]) {
                o = (char*)argv[i+1];   /* -o */
                vars[5] = false;
            }
            if (!strcmp("-N", argv[i]) && vars[6]) {
                n = (char*)argv[i+1];  /* -N */
                vars[6] = false;
            }
            if (!strcmp("-R", argv[i]) && vars[7]) {
                r = (char*)argv[i+1];   /* -R */
                vars[7] = false;
            }
        }

        if(d==NULL || q==NULL){
            cout << "You must run the program with parameters(REQUIRED): –d <input file> –q <query file>" << endl;
            cout << "With additional parameters: –k <int> -L <int> -ο <output file> -Ν <number of nearest> -R <radius>" << endl;
            return(0);
        }
        else{
            if (k!=NULL) K = atoi(k);
            if (m!=NULL) Max = atoi(m);
            if (n!=NULL) N = atoi(n);
            if (pr!=NULL) probes = atoi(pr);
            if (r!=NULL) R = atof(r);

            unsigned int magicNumber = 0,numberOfImages = 0,numberOfRows = 0,numberOfColumns = 0, img=0;
            unsigned int numOfpixels;

            //Open train file
            fstream trainInput(d);
            if(!trainInput.is_open()){
                cerr<<"Failed to open input data."<<endl;
                return(0);
            }
            trainInput.read((char*)&magicNumber, 4);
            trainInput.read((char*)&numberOfImages, 4);
            trainInput.read((char*)&numberOfRows, 4);
            trainInput.read((char*)&numberOfColumns, 4);

            //Convert intergers from Big Endian to Little Endian
            magicNumber = SWAP_INT32(magicNumber);
            numberOfImages = SWAP_INT32(numberOfImages);
            numberOfRows = SWAP_INT32(numberOfRows);
            numberOfColumns = SWAP_INT32(numberOfColumns);
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

                int bucketsNumber = trainSet.getNumberOfImages()/16; //pow(2,K);

                // int W = FindW(img, &trainSet);
                // cout << "W is " << W << endl;
                int W = 40000;
                HashFunction** hashFamily = new HashFunction*[trainSet.getNumberOfPixels()];
                for(int i = 0; i < trainSet.getNumberOfPixels();i++){
                    hashFamily[i] = NULL;
                }
                Projection *projection = new Projection(trainSet.getNumberOfPixels(),bucketsNumber, K,W, hashFamily);
                for(int j=0; j<trainSet.getNumberOfImages(); j++){
                    unsigned int g_hash = (unsigned int)(projection->ghash(trainSet.imageAt(j)));
                    projection->getBucketArray()[g_hash%bucketsNumber]->addImage(j,g_hash,trainSet.imageAt(j));
                }

                ofstream outputf(o);
                if (!outputf.is_open()){
                    cerr<<"Failed to open output data."<<endl;
                    return 0;
                }
                double cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime;
                for(int i=0; i<numberOfImages; i++){
                    vector<Neighbor> ANNneighbors;
                    vector<Neighbor> RNGneighbors;
                    vector<double> ANNtrueDist;
                    vector<double> RNGtrueDist;
                    hyperCubeSearch(ANNneighbors, RNGneighbors, ANNtrueDist, RNGtrueDist, cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime, true, R, N, probes, i, querySet.imageAt(i),&trainSet, projection);
                    cubeOutput(outputf, i,  N, ANNneighbors, RNGneighbors, ANNtrueDist, RNGtrueDist,cubeAnnTime, cubeRngTime, trueAnnTime, trueRngTime);
                }
                outputf.close();

                for(int i=0; i<trainSet.getNumberOfPixels(); i++){
                    if(hashFamily[i]!=NULL){
                        delete hashFamily[i];
                    }
                }
                delete[] hashFamily;
                delete projection;

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
