#pragma once

#include <MPI/OPP_MPI_base.h>

namespace OPP {
    // Defines the MPI namespace inside OPP ...
    namespace MPI {

        /**
         * @brief Represents a Ring topology for a given communicator
         * 
         * All the process into the given communicator participates to the ring, using their rank.
         * 
         * A processor may send a message to the next process into the ring, and so receive only from the previous one.
         */
        class Ring 
        {
        protected:
            const Communicator communicator;

            const int next;
            const int previous;

        public:

            int getRank() const {
                return communicator.rank;
            }

            int getSize() const {
                return communicator.size;
            }

            int getPrevious() const {
                return previous;
            }

            int getNext() const {
                return next;
            }

            MPI_Comm getComm() const {
                return communicator.communicator;
            }

            Ring(const MPI_Comm communicator) :
                communicator(communicator), 
                next((getRank()+1) % getSize()), 
                previous((getRank()+getSize()-1) % getSize())
            {}

            Ring() = delete;
            Ring(const Ring&) = default;
            ~Ring() = default;
            Ring& operator=(const Ring&) = delete;

            void Send(void const*const message, const int length, const MPI_Datatype datatype, const int tag=0) const
            {
                checkMPIErrors( MPI_Send(message, length, datatype, next, tag, getComm()) );
            }

            MPI_Request AsyncSend(void const*const message, const int length, const MPI_Datatype datatype, const int tag=0) const
            {
                MPI_Request request;
                checkMPIErrors( MPI_Isend(message, length, datatype, next, tag, getComm(), &request) );
                return request;
            }

            MPI_Status Recv(void *const message, const int length, const MPI_Datatype datatype, const int tag=0) const
            {
                MPI_Status status;
                checkMPIErrors( MPI_Recv(message, length, datatype, previous, tag, getComm(), &status) );
                return status;
            }

            MPI_Request AsyncRecv(void *const message, const int length, const MPI_Datatype datatype, const int tag=0) const
            {
                MPI_Request request;
                checkMPIErrors( MPI_Irecv(message, length, datatype, previous, tag, getComm(), &request) );
                return request;
            }

        };


        /**
         * @brief Represents a Bidirectional Ring topology for a given communicator
         * 
         * All the process into the given communicator participates to the ring, using their rank.
         * 
         * A processor may send and receive a message only to the next or previous processes into the ring.
         */
        class BidirRing : protected Ring {

        public:
            int getRank() const {
                return communicator.rank;
            }

            int getSize() const {
                return communicator.size;
            }

            MPI_Comm getComm() const {
                return communicator.communicator;
            }

            int getPrevious() const {
                return previous;
            }

            int getNext() const {
                return next;
            }

            enum Direction { PREVIOUS=-1, NEXT=1 };

            BidirRing(const MPI_Comm communicator) : Ring(communicator)
            {}

            BidirRing() = delete;
            BidirRing(const BidirRing&) = default;
            ~BidirRing() = default;
            BidirRing& operator=(const BidirRing&) = delete;

            void Send(void const*const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const int dest = direction == NEXT ? next : previous;
                checkMPIErrors( MPI_Send(message, length, datatype, dest, tag, getComm()) );
            }

            MPI_Request AsyncSend(void const*const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const int dest = direction == NEXT ? next : previous;
                MPI_Request request;
                checkMPIErrors( MPI_Isend(message, length, datatype, dest, tag, getComm(), &request) );
                return request;
            }

            MPI_Status Recv(void *const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const int src = direction == NEXT ? next : previous;                
                MPI_Status status;
                checkMPIErrors( MPI_Recv(message, length, datatype, src, tag, getComm(), &status) );
                return status;
            }

            MPI_Request AsyncRecv(void *const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const int src = direction == NEXT ? next : previous;                
                MPI_Request request;
                checkMPIErrors( MPI_Irecv(message, length, datatype, src, tag, getComm(), &request) );
                return request;
            }

        };

    }
}