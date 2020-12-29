#include "mpi_send_recv.h"

ISend::ISend(const ArrayHandle<uint8_t> &data,
             int count,
             MPI_Datatype send_type,
             int dest,
             int tag,
             MPI_Comm comm)
    : data(data)
{
    start_send(count, send_type, dest, tag, comm);
}

bool ISend::complete()
{
    if (request == MPI_REQUEST_NULL) {
        return true;
    }
    int flag = 0;
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    return flag != 0;
}

void ISend::start_send(int count, MPI_Datatype send_type, int dest, int tag, MPI_Comm comm)
{
    this->count = count;
    this->send_type = send_type;
    this->dest = dest;
    this->tag = tag;
    this->comm = comm;
    MPI_Isend(data->data(), count, send_type, dest, tag, comm, &request);
}

IRecv::IRecv(ArrayHandle<uint8_t> &data,
             int count,
             MPI_Datatype recv_type,
             int src,
             int tag,
             MPI_Comm comm)
    : data(data)
{
    start_recv(count, recv_type, src, tag, comm);
}

bool IRecv::complete()
{
    if (request == MPI_REQUEST_NULL) {
        return true;
    }
    int flag = 0;
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    return flag != 0;
}

void IRecv::start_recv(int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm)
{
    this->count = count;
    this->recv_type = recv_type;
    this->src = src;
    this->tag = tag;
    this->comm = comm;
    MPI_Irecv(data->data(), count, recv_type, src, tag, comm, &request);
}
