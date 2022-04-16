#pragma once
#include <MPI/OPP_MPI_base.h>
#include <MPI/OPP_MPI_Ring.h>
#include <cmath>

namespace OPP
{
    namespace MPI
    {
        class Torus
        {
        protected:
            const Communicator communicator;
            const BidirRing columnRing;
            const BidirRing rowRing;

            Torus(Communicator &communicator, const BidirRing &columnRing, const BidirRing &rowRing)
                : communicator(communicator), columnRing(columnRing), rowRing(rowRing)
            {
            }

        public:
            static Torus build(const MPI_Comm &comm)
            {
                Communicator communicator(comm);
                const int P = int(round(sqrt(double(communicator.size))));
                if (P * P != communicator.size)
                {
                    abortOnMPIErrors("Number of processes must be a square number", -1);
                }
                const int column = communicator.rank % P;
                const int row = communicator.rank / P;
                MPI_Comm commColumn, commRow;
                checkMPIErrors( 
                    MPI_Comm_split(comm, column, row, &commColumn)
                );
                checkMPIErrors( 
                    MPI_Comm_split(comm, row, column, &commRow) 
                );
                return Torus(communicator, BidirRing(commColumn), BidirRing(commRow));
            }

            Torus() = delete;
            Torus(const Torus&) = default;
            Torus operator=(const Torus&) = delete;
            ~Torus() = default;

            enum Direction { WEST=-1, NORTH=-2, EAST=1, SOUTH=2};
            
            BidirRing getColumnRing() const {
                return columnRing;
            }

            BidirRing getRowRing() const {
                return rowRing;
            }   
            
        protected:
            BidirRing getRing(const Direction direction) const 
            {
                if( direction == WEST || direction == EAST )
                    return rowRing;
                return columnRing;
            }

            BidirRing::Direction getRingDirection(const Direction direction) const
            {
                if( direction == WEST || direction == NORTH )
                    return BidirRing::Direction::PREVIOUS;
                return BidirRing::Direction::NEXT;            
            }

        public:
            void Send(void const*const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0)  const
            {
                const auto ring = getRing(direction);
                const auto ringDirection = getRingDirection(direction);
                ring.Send(message, length, datatype, ringDirection, tag);
            }

            MPI_Request AsyncSend(void const*const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const auto ring = getRing(direction);
                const auto ringDirection = getRingDirection(direction);
                ring.AsyncSend(message, length, datatype, ringDirection, tag);
            }

            MPI_Status Recv(void *const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const auto ring = getRing(direction);
                const auto ringDirection = getRingDirection(direction);
                ring.Recv(message, length, datatype, ringDirection, tag);
            }

            MPI_Request AsyncRecv(void *const message, const int length, const MPI_Datatype datatype, const Direction direction, const int tag=0) const
            {
                const auto ring = getRing(direction);
                const auto ringDirection = getRingDirection(direction);
                ring.AsyncRecv(message, length, datatype, ringDirection, tag);
            }         
        };
    }
}