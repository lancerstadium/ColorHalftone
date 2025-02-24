#ifndef SR_LUT_DATA_H
#define SR_LUT_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 使用枚举类型表示数据类型
typedef enum {
    TYPE_FLOAT32 = 1,
    TYPE_INT32 = 2,
    TYPE_FLOAT64 = 3,
    TYPE_INT64 = 4,
    TYPE_INT16 = 5,
    TYPE_UINT8 = 6,
    TYPE_COMPLEX64 = 7,
    TYPE_COMPLEX128 = 8,
    TYPE_BOOL = 9,
    TYPE_UINT32 = 10,
    TYPE_UINT64 = 11,
    TYPE_INT8 = 12  // 添加 int8 类型
} TensorType;

char* TensorType_str(TensorType dtype);

// 张量：包含维度数、维度数据和加载的数据
typedef struct {
    int ndim;       // 维度数
    int *shape;     // 维度数据
    void *data;     // 加载的张量数据
    TensorType dtype; // 数据类型
} TensorData;

TensorData* TensorData_load(const char *bin_file);
void TensorData_free(TensorData *tensor);
size_t TensorData_size(const TensorData *tensor);
void TensorData_test(const char *bin_file);

#endif //SR_LUT_DATA_H