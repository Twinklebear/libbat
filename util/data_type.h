#pragma once

#include <string>

enum DTYPE {
    UNKNOWN,
    INT_8,
    UINT_8,
    INT_16,
    UINT_16,
    INT_32,
    UINT_32,
    INT_64,
    UINT_64,
    FLOAT_32,
    FLOAT_64,
    VEC2_I8,
    VEC2_U8,
    VEC2_I16,
    VEC2_U16,
    VEC2_I32,
    VEC2_U32,
    VEC2_FLOAT,
    VEC2_DOUBLE,
    VEC3_I8,
    VEC3_U8,
    VEC3_I16,
    VEC3_U16,
    VEC3_I32,
    VEC3_U32,
    VEC3_FLOAT,
    VEC3_DOUBLE,
    VEC4_I8,
    VEC4_U8,
    VEC4_I16,
    VEC4_U16,
    VEC4_I32,
    VEC4_U32,
    VEC4_FLOAT,
    VEC4_DOUBLE,
    MAT2_I8,
    MAT2_U8,
    MAT2_I16,
    MAT2_U16,
    MAT2_I32,
    MAT2_U32,
    MAT2_FLOAT,
    MAT2_DOUBLE,
    MAT3_I8,
    MAT3_U8,
    MAT3_I16,
    MAT3_U16,
    MAT3_I32,
    MAT3_U32,
    MAT3_FLOAT,
    MAT3_DOUBLE,
    MAT4_I8,
    MAT4_U8,
    MAT4_I16,
    MAT4_U16,
    MAT4_I32,
    MAT4_U32,
    MAT4_FLOAT,
    MAT4_DOUBLE,
};

std::string print_data_type(DTYPE type);

size_t dtype_stride(DTYPE type);

size_t dtype_components(DTYPE type);

template <typename T>
struct GetDType {
};

template <>
struct GetDType<int8_t> {
    const static DTYPE type = INT_8;
};

template <>
struct GetDType<uint8_t> {
    const static DTYPE type = UINT_8;
};

template <>
struct GetDType<int16_t> {
    const static DTYPE type = INT_16;
};

template <>
struct GetDType<uint16_t> {
    const static DTYPE type = UINT_16;
};

template <>
struct GetDType<int32_t> {
    const static DTYPE type = INT_32;
};

template <>
struct GetDType<uint32_t> {
    const static DTYPE type = UINT_32;
};

template <>
struct GetDType<int64_t> {
    const static DTYPE type = INT_64;
};

template <>
struct GetDType<uint64_t> {
    const static DTYPE type = UINT_64;
};

template <>
struct GetDType<float> {
    const static DTYPE type = FLOAT_32;
};

template <>
struct GetDType<double> {
    const static DTYPE type = FLOAT_64;
};
