#include "byte_cursor.h"

ByteCursor::ByteCursor(uint8_t *ptr) : ptr(ptr) {}

uint8_t *ByteCursor::position()
{
    return ptr + offset;
}

void ByteCursor::reset()
{
    offset = 0;
}

void ByteCursor::advance(size_t nbytes)
{
    offset += nbytes;
}
