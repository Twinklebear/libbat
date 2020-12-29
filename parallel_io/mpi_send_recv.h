#pragma once

#include <memory>
#include <mpi.h>
#include "abstract_array.h"
#include "borrowed_array.h"
#include "owned_array.h"

struct ISend {
    ArrayHandle<uint8_t> data;
    MPI_Request request = MPI_REQUEST_NULL;

    int count;
    MPI_Datatype send_type;
    int dest;
    int tag;
    MPI_Comm comm;

    ISend() = default;

    template <typename T>
    ISend(const T *t,
          int count,
          MPI_Datatype send_type,
          int dest,
          int tag,
          MPI_Comm comm,
          bool shared_data);

    ISend(const ArrayHandle<uint8_t> &data,
          int count,
          MPI_Datatype send_type,
          int dest,
          int tag,
          MPI_Comm comm);

    bool complete();

private:
    void start_send(int count, MPI_Datatype send_type, int dest, int tag, MPI_Comm comm);
};

struct IRecv {
    ArrayHandle<uint8_t> data;
    MPI_Request request = MPI_REQUEST_NULL;

    int count;
    MPI_Datatype recv_type;
    int src;
    int tag;
    MPI_Comm comm;

    IRecv() = default;

    template <typename T>
    static IRecv recv(int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm);

    template <typename T>
    IRecv(T *t, int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm);

    IRecv(ArrayHandle<uint8_t> &data,
          int count,
          MPI_Datatype recv_type,
          int src,
          int tag,
          MPI_Comm comm);

    bool complete();

private:
    void start_recv(int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm);
};

template <typename T>
ISend::ISend(const T *t,
             int count,
             MPI_Datatype send_type,
             int dest,
             int tag,
             MPI_Comm comm,
             bool shared_data)
{
    if (shared_data) {
        data = std::make_shared<BorrowedArray<uint8_t>>(reinterpret_cast<const uint8_t *>(t),
                                                        count * sizeof(T));
    } else {
        data = std::make_shared<OwnedArray<uint8_t>>(reinterpret_cast<const uint8_t *>(t),
                                                     count * sizeof(T));
    }
    start_send(count, send_type, dest, tag, comm);
}

template <typename T>
IRecv IRecv::recv(int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm)
{
    auto data = std::dynamic_pointer_cast<AbstractArray<uint8_t>>(
        std::make_shared<OwnedArray<uint8_t>>(count * sizeof(T)));
    return IRecv(data, count, recv_type, src, tag, comm);
}

template <typename T>
IRecv::IRecv(T *t, int count, MPI_Datatype recv_type, int src, int tag, MPI_Comm comm)
    : data(std::make_shared<BorrowedArray<uint8_t>>(reinterpret_cast<uint8_t *>(t),
                                                    count * sizeof(T)))
{
    start_recv(count, recv_type, src, tag, comm);
}
