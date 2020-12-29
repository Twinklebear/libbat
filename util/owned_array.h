#pragma once

#include <array>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "abstract_array.h"

template <typename T>
class OwnedArray : public AbstractArray<T> {
public:
    std::vector<T> array;

    OwnedArray() = default;

    OwnedArray(const std::vector<T> &array) : array(array) {}

    OwnedArray(std::vector<T> &&array) : array(array) {}

    OwnedArray(const T *t, size_t size) : array(t, t + size) {}

    explicit OwnedArray(size_t size)
    {
        resize(size);
    }

    virtual ~OwnedArray() = default;

    const T &at(const size_t i) const
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing OwnedArray out of bounds");
        }
#endif
        return array[i];
    }

    T &at(const size_t i)
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing OwnedArray out of bounds");
        }
#endif
        return array[i];
    }

    const T &operator[](const size_t i) const override
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing OwnedArray out of bounds");
        }
#endif
        return array[i];
    }

    T &operator[](const size_t i)
    {
#ifdef DEBUG
        if (i > size()) {
            throw std::runtime_error("Accessing OwnedArray out of bounds");
        }
#endif
        return array[i];
    }

    const T *data() const override
    {
        return array.data();
    }

    T *data() override
    {
        return array.data();
    }

    size_t size() const override
    {
        return array.size();
    }

    size_t size_bytes() const override
    {
        return array.size() * sizeof(T);
    }

    const T *cbegin() const override
    {
        return array.data();
    }

    T *begin()
    {
        return array.data();
    }

    T *end()
    {
        return begin() + size();
    }

    void clear()
    {
        array.clear();
    }

    void reserve(const size_t n)
    {
        array.reserve(n);
    }

    void resize(const size_t n, const T &t = T{})
    {
        if (n > 0) {
            array.resize(n, t);
        }
    }

    void push_back(const T &t)
    {
        array.push_back(t);
    }

    template <typename = std::enable_if<std::is_same<T, uint8_t>::value>>
    void push_back(const void *t, const size_t bytes)
    {
        const uint8_t *b = reinterpret_cast<const uint8_t *>(t);
        std::copy(b, b + bytes, std::back_inserter(array));
    }
};

template <typename T>
using OwnedArrayHandle = std::shared_ptr<OwnedArray<T>>;

