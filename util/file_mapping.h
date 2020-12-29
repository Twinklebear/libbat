#pragma once

#include <memory>
#include <string>
#include <vector>
#include "abstract_array.h"

#ifdef _WIN32
#include <windows.h>
#endif

class FileData {
public:
    virtual ~FileData() {}

    virtual const uint8_t *data() const = 0;

    virtual uint8_t *data() = 0;

    virtual size_t nbytes() const = 0;
};

class FileMapping : public FileData {
    void *mapping;
    size_t num_bytes;
#ifdef _WIN32
    HANDLE file;
    HANDLE mapping_handle;
#else
    int file;
#endif

public:
    // Map the file into memory
    FileMapping(const std::string &fname);

    FileMapping(FileMapping &&fm);

    ~FileMapping();

    FileMapping &operator=(FileMapping &&fm);

    FileMapping(const FileMapping &) = delete;

    FileMapping &operator=(const FileMapping &) = delete;

    const uint8_t *data() const override;

    uint8_t *data() override;

    size_t nbytes() const override;
};

class FileBuffer : public FileData {
    std::vector<uint8_t> buffer;

public:
    // Read the file into memory
    FileBuffer(const std::string &fname);

    FileBuffer(const FileBuffer &) = delete;

    FileBuffer &operator=(const FileBuffer &) = delete;

    const uint8_t *data() const override;

    uint8_t *data() override;

    size_t nbytes() const override;
};

template <typename T>
class FileView : public AbstractArray<T> {
    std::shared_ptr<FileData> file;
    T *ptr = nullptr;
    size_t count = 0;

public:
    FileView() = default;

    /* Create a typed view into the file. The offset is in bytes, the count in
     * number of elements of T in the view.
     */
    FileView(const std::shared_ptr<FileData> &f, size_t offset, size_t count)
        : file(f), ptr(reinterpret_cast<T *>(file->data() + offset)), count(count)
    {
    }

    const T &operator[](const size_t i) const override
    {
        return ptr[i];
    }

    const T *data() const override
    {
        return ptr;
    }

    T *data() override
    {
        return ptr;
    }

    size_t size() const override
    {
        return count;
    }

    size_t size_bytes() const override
    {
        return count * sizeof(T);
    }

    const T *cbegin() const override
    {
        return ptr;
    }

    const T *cend() const override
    {
        return ptr + count;
    }
};
