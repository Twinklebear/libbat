#pragma once

#include <stdexcept>
#include "abstract_array.h"

template <typename T>
class FixedArray : public AbstractArray<T> {
    std::shared_ptr<T> array = nullptr;
    size_t sz = 0;

public:
    FixedArray() = default;

    FixedArray(size_t sz)
        : array(std::shared_ptr<T>(new T[sz], std::default_delete<T[]>())), sz(sz)
    {
    }

    virtual ~FixedArray() = default;

    const T &at(const size_t i) const
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing FixedArray out of bounds");
        }
#endif
        return array.get()[i];
    }

    T &at(const size_t i)
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing FixedArray out of bounds");
        }
#endif
        return array.get()[i];
    }

    const T &operator[](const size_t i) const override
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing FixedArray out of bounds");
        }
#endif
        return array.get()[i];
    }

    T &operator[](const size_t i)
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing FixedArray out of bounds");
        }
#endif
        return array.get()[i];
    }

    const T *data() const override
    {
        return array.get();
    }

    T *data() override
    {
        return array.get();
    }

    size_t size() const override
    {
        return sz;
    }

    size_t size_bytes() const override
    {
        return sz * sizeof(T);
    }

    const T *cbegin() const override
    {
        return array.get();
    }

    T *begin()
    {
        return array.get();
    }

    T *end()
    {
        return begin() + size();
    }
};
