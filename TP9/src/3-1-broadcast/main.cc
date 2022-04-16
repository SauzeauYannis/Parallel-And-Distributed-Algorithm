#include <sstream>
#include <iostream>
#include <MPI/OPP_MPI.h>
#include <cstring>
#include <algorithm> // for std::max/min
#include <utils/chronoCPU.h>
#include "Broadcast.h"

#ifndef WIN32
using namespace std;
#endif

namespace 
{
    void parse_command_line(int argc, char**argv, int& e) 
    {

        for(int i=0; i<argc-1; ++i) {
            if( !strcmp("-e", argv[i]) ) {
                std::istringstream is(argv[i+1]);
                int value = e;
                try {
                    if( is >> value ) {
                        e = max(1, min(value, 29));
                        i++;
                    }
                } catch(...) {}
            }
            else if( !strcmp("-h", argv[i])) {
                std::cout<<"Options:"<<std::endl;
                std::cout<<"\t-e <e>: log2(n), for vector of size n."<<std::endl;
                std::cout<<"\t-h : display this help."<<std::endl;
                exit( 0 );
            }
        }
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
    
    int e = 10; // by default 2^10, so 1024
    ::parse_command_line(argc, argv, e);

    const int N = 1<<e;

    for(int k=0; k<ring.getSize(); ++k) 
    {
        int*const array = ::init_array(N, ring.getRank());
    
        
        if( k == ring.getRank() )
            std::cout<<"Broadcast from "<<k<<" an array of "<<N<<" integers ..."<<std::endl;
        ChronoCPU chr;
        MPI_Barrier( MPI_COMM_WORLD );
        chr.start();
            Broadcast(k, array, N);
            MPI_Barrier( MPI_COMM_WORLD );
        chr.stop();
        std::cout <<"[" << ring.getRank() << "] Done in " << chr.elapsedTime() << "ms."<<std::endl;

        ::display_errors(array, N, k, ring.getRank());

        delete array;
    }

    OPP::MPI::Initializer::close();

    return 0;
}
