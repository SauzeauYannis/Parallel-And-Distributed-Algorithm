#include <cstring>
#include <sstream>
#include <iostream>
#include <MPI/OPP_MPI.h>
//#include <algorithm> // for std::max/min
#include <utils/chronoCPU.h>
#include "Broadcast.h"

#ifndef WIN32
using namespace std;
#endif

namespace 
{
    void parse_command_line(int argc, char**argv, int& en, int &em) 
    {

        for(int i=0; i<argc-1; ++i) {
            if( !strcmp("-en", argv[i]) ) {
                std::istringstream is(argv[i+1]);
                int value = en;
                try {
                    if( is >> value ) {
                        en = max(1, min(value,29));
                        i++;
                    }
                } catch(...) {}
            }
            else if( !strcmp("-em", argv[i]) ) {
                std::istringstream is(argv[i+1]);
                int value = em;
                try {
                    if( is >> value ) {
                        em = max(1, min(value,29));
                        i++;
                    }
                } catch(...) {}
            }
            else if( !strcmp("-h", argv[i])) {
                std::cout<<"Options:"<<std::endl;
                std::cout<<"\t-en <en>: log2(n), for vector of size n."<<std::endl;
                std::cout<<"\t-em <em>: log2(m), for packet (on pipeline) of size m."<<std::endl;
                std::cout<<"\t-h : display this help."<<std::endl;
                exit( 0 );
            }
        }
    }

    int calculateN(const int proposed, const int M) {
        const int remainder = proposed%M;
        if( remainder == 0 )
            return proposed;
        return proposed + (M-remainder);
    }

    int *init_array(const int N, const int rank) 
    {
        int*array = new int[N];
        for(int i=0; i<N; ++i) 
            array[i] = rank;
        return array;
    }

    void display_errors(const int*const array, const int N, const int k, const int rank) 
    {
        int errors = 0;
        for(int i=0; i<N; ++i) 
        {
            if(array[i] != k) {
                std::cerr<<"[" << rank << "] Error for value "<<i<<" (receive "<<array[i]<<")"<<std::endl;
                ++ errors;
            }
            if( errors == 10 ) {
                std::cerr<<"[" << rank << "] ... and so on"<<std::endl;
                break;
            }
        }
        if( errors == 0 )
            std::cout << "[" << rank << "] Good, your broadcast seems to work ;-)"<<std::endl;
    }
}

// entry function
int main(int argc, char**argv)
{
    OPP::MPI::Initializer::init(&argc, &argv);

    OPP::MPI::Ring ring(MPI_COMM_WORLD);
    
    int en = 10; // by default 2^10, so 1024
    int em = 10;
    ::parse_command_line(argc, argv, en, em);
    if( em >= (en>>1) ) 
        em = en >> 2;
    const int M = 1<<em;
    const int N = calculateN(1<<en, M);

    for(int k=0; k<ring.getSize(); ++k) 
    {
        int*const array = ::init_array(N, ring.getRank());
            
        if( k == ring.getRank() )
            std::cout<<"Broadcast from "<<k<<" an array of "<<N<<" integers ..."<<std::endl;
        ChronoCPU chr;
        MPI_Barrier( ring.getComm() );
        chr.start();
            Broadcast(k, array, N, M);
            MPI_Barrier( ring.getComm() );
        chr.stop();
        std::cout <<"[" << ring.getRank() << "] Done in " << chr.elapsedTime() << "ms."<<std::endl;

        ::display_errors(array, N, k, ring.getRank());

        delete array;
    }

    OPP::MPI::Initializer::close();
    return 0;
}
