#pragma once

#include <mpi.h>
#include <iostream>

#define abortOnMPIErrors(msg, error_code) do {\
    std::cerr << "Fail using MPI with error message: " << msg << std::endl;\
    MPI_Abort(MPI_COMM_WORLD, error_code);\
} while(0)

#define checkMPIErrors(cmd) do {\
    const auto error_code = cmd; \
    if( error_code != MPI_SUCCESS ) { \
        std::cerr << "Error received doing " << #cmd << std::endl;\
        int size = MPI_MAX_ERROR_STRING; \
        char error_message[MPI_MAX_ERROR_STRING]; \
        MPI_Error_string(error_code, error_message, &size); \
        abortOnMPIErrors(error_message, error_code);\
    } \
} while(0)

namespace OPP {
    // Defines the MPI namespace inside OPP ...
    namespace MPI {

        /** Utility class to start and stop the MPI machinery */
        struct Initializer 
        {
            /**
             * @brief Start or initialize the MPI system. This can be done only once.
             * @param argc pointer to the number of command line arguments 
             * @param argv pointer to the array of command line arguments
             * @param LEVEL behavior regarding threads (MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, MPI_THREAD_MULTIPLE)
             * @return the thread level
             */
            static int init(int *argc=nullptr, char***argv=nullptr, const int LEVEL=MPI_THREAD_SINGLE) {
                int isInitialized, level;
                checkMPIErrors( MPI_Initialized( &isInitialized ) );
                if( !isInitialized )
                    checkMPIErrors( MPI_Init_thread(argc, argv, LEVEL, &level) );
                else 
                    checkMPIErrors( MPI_Query_thread(&level) );
                return level;
            }

            /**
             * @brief Stop or close the MPI system. This can be done only once.
             */
            static void close() {
                int isFinalized;
                checkMPIErrors( MPI_Finalized( &isFinalized ) );
                if( !isFinalized )
                    checkMPIErrors( MPI_Finalize() );
            }
        };

        /**
         * @brief Represents a communicator, its size and the rank of the current process.         * 
         */
        struct Communicator {
            const MPI_Comm communicator;
            int rank;
            int size;

            /**
             * @brief Construct a new Communicator object
             * 
             * @param communicator the represented MPI_Comm 
             */
            Communicator(const MPI_Comm& communicator) : communicator(communicator) 
            {
                checkMPIErrors( MPI_Comm_size(communicator, &size) );

                checkMPIErrors( MPI_Comm_rank(communicator, &rank) );
            }

            Communicator() = delete;
            Communicator(const Communicator&) = default;
            Communicator& operator=(const Communicator&) = default;
        };

    }
}
