#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

struct ByteCursor {
    uint8_t *ptr = nullptr;
    size_t offset = 0;

    ByteCursor(uint8_t *ptr);

    ByteCursor() = default;

    uint8_t *position();

    template <typename T>
    void write(const T *t, const size_t count);

    template <typename T>
    void write(const T &t);

    template <typename T>
    void read(T *t, const size_t count);

    template <typename T>
    void read(T &t);

    void reset();

    void advance(size_t nbytes); 
};

template <typename T>
void ByteCursor::write(const T *t, const size_t count)
{
    std::memcpy(ptr + offset, t, count * sizeof(T));
    offset += count * sizeof(T);
}

template <typename T>
void ByteCursor::write(const T &t)
{
    std::memcpy(ptr + offset, &t, sizeof(T));
    offset += sizeof(T);
}

template <typename T>
void ByteCursor::read(T *t, const size_t count)
{
    std::memcpy(t, ptr + offset, count * sizeof(T));
    offset += count * sizeof(T);
}

template <typename T>
void ByteCursor::read(T &t)
{
    std::memcpy(&t, ptr + offset, sizeof(T));
    offset += sizeof(T);
}
