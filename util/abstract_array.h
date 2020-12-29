#pragma once

#include <memory>

template <typename T>
class AbstractArray {
public:
    virtual ~AbstractArray() = default;

    virtual const T &at(const size_t i) const
    {
        return *(data() + i);
    }

    virtual const T &operator[](const size_t i) const = 0;

    virtual const T *data() const = 0;

    virtual T *data() = 0;

    virtual size_t size() const = 0;

    virtual size_t size_bytes() const = 0;

    virtual const T *cbegin() const = 0;

    virtual const T *cend() const
    {
        return cbegin() + size();
    }
};

template <typename T>
using ArrayHandle = std::shared_ptr<AbstractArray<T>>;
