#include "data.h"


char* TensorType_str(TensorType dtype) {
    switch (dtype) {
        case TYPE_FLOAT32: return "float32";
        case TYPE_INT32: return "int32";
        case TYPE_FLOAT64: return "float64";
        case TYPE_INT64: return "int64";
        case TYPE_INT16: return "int16";
        case TYPE_UINT8: return "uint8";
        case TYPE_COMPLEX64: return "complex64";
        case TYPE_COMPLEX128: return "complex128";
        case TYPE_BOOL: return "bool";
        case TYPE_UINT32: return "uint32";
        case TYPE_UINT64: return "uint64";
        case TYPE_INT8: return "int8";
        default: return "unknown";
    }
}

size_t TensorType_size(TensorType dtype) {
    switch (dtype) {
        case TYPE_FLOAT32: return sizeof(float);
        case TYPE_INT32: return sizeof(int32_t);
        case TYPE_FLOAT64: return sizeof(double);
        case TYPE_INT64: return sizeof(int64_t);
        case TYPE_INT16: return sizeof(int16_t);
        case TYPE_UINT8: return sizeof(uint8_t);
        case TYPE_COMPLEX64: return 8;
        case TYPE_COMPLEX128: return 16;
        case TYPE_BOOL: return 4;
        case TYPE_UINT32: return sizeof(uint32_t);
        case TYPE_UINT64: return sizeof(uint64_t);
        case TYPE_INT8: return sizeof(int8_t);
        default: return 0;
    }
}

// 根据数据类型读取相应的数据
void* _read_bin(FILE *file, TensorType dtype, size_t num_elements) {
    void *data = NULL;
    size_t res = 0;
    switch (dtype) {
        case TYPE_FLOAT32:
            data = malloc(num_elements * sizeof(float));
            res = fread(data, sizeof(float), num_elements, file);
            break;
        case TYPE_INT32:
            data = malloc(num_elements * sizeof(int32_t));
            res = fread(data, sizeof(int32_t), num_elements, file);
            break;
        case TYPE_FLOAT64:
            data = malloc(num_elements * sizeof(double));
            res = fread(data, sizeof(double), num_elements, file);
            break;
        case TYPE_INT64:
            data = malloc(num_elements * sizeof(int64_t));
            res = fread(data, sizeof(int64_t), num_elements, file);
            break;
        case TYPE_INT16:
            data = malloc(num_elements * sizeof(int16_t));
            res = fread(data, sizeof(int16_t), num_elements, file);
            break;
        case TYPE_UINT8:
            data = malloc(num_elements * sizeof(uint8_t));
            res = fread(data, sizeof(uint8_t), num_elements, file);
            break;
        case TYPE_COMPLEX64:
            // 假设每个复数用两个float表示（实部和虚部）
            data = malloc(num_elements * sizeof(float) * 2);
            res = fread(data, sizeof(float) * 2, num_elements, file);
            break;
        case TYPE_COMPLEX128:
            // 假设每个复数用两个double表示（实部和虚部）
            data = malloc(num_elements * sizeof(double) * 2);
            res = fread(data, sizeof(double) * 2, num_elements, file);
            break;
        case TYPE_BOOL:
            data = malloc(num_elements * sizeof(uint8_t));
            res = fread(data, sizeof(uint8_t), num_elements, file);
            break;
        case TYPE_UINT32:
            data = malloc(num_elements * sizeof(uint32_t));
            res = fread(data, sizeof(uint32_t), num_elements, file);
            break;
        case TYPE_UINT64:
            data = malloc(num_elements * sizeof(uint64_t));
            res = fread(data, sizeof(uint64_t), num_elements, file);
            break;
        case TYPE_INT8:
            data = malloc(num_elements * sizeof(int8_t));
            res = fread(data, sizeof(int8_t), num_elements, file);
            break;
        default:
            fprintf(stderr, "Unsupported data type\n");
            return NULL;
    }

    return data;
}

// 加载二进制文件的函数
TensorData* TensorData_load(const char *bin_file) {
    FILE *file = fopen(bin_file, "rb");
    if (!file) {
        fprintf(stderr, "Unable to open file: %s\n", bin_file);
        exit(EXIT_FAILURE);
        return NULL;
    }
    size_t res = 0;
    TensorData *tensor = (TensorData *)malloc(sizeof(TensorData));

    // 读取数据类型（4字节）
    int32_t dtype_code;
    res = fread(&dtype_code, sizeof(int32_t), 1, file);
    tensor->dtype = (TensorType)dtype_code;

    // 读取维度数（4字节）
    res = fread(&tensor->ndim, sizeof(int32_t), 1, file);

    // 读取每个维度的大小
    tensor->shape = (int32_t *)malloc(tensor->ndim * sizeof(int32_t));
    res = fread(tensor->shape, sizeof(int32_t), tensor->ndim, file);

    // 计算总元素数
    size_t num_elements = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        num_elements *= tensor->shape[i];
    }

    // 根据数据类型读取数据
    tensor->data = _read_bin(file, tensor->dtype, num_elements);

    fclose(file);
    return tensor;
}

size_t inline TensorData_size(const TensorData *tensor) {
    if(!tensor)
        return 0;
    size_t size = 1;
    for(int i = 0; i < tensor->ndim; i++) {
        size *= (tensor->shape[i] * TensorType_size(tensor->dtype));
    }
    return size;
}



// 释放TensorData内存
void TensorData_free(TensorData *tensor) {
    if (tensor) {
        free(tensor->shape); tensor->shape = NULL;
        free(tensor->data);  tensor->data = NULL;
        free(tensor);        tensor = NULL;
    }
}

void TensorData_test(const char *bin_file) {
    TensorData *tensor = TensorData_load(bin_file);
    if (tensor) {
        printf("Data type: %d\n", tensor->dtype);
        printf("Dimensions: %d\n", tensor->ndim);
        printf("Shape: ");
        for (int i = 0; i < tensor->ndim; i++) {
            printf("%d ", tensor->shape[i]);
        }
        printf("\n");

        // 根据 dtype 输出数据
        if (tensor->dtype == TYPE_FLOAT32) {
            float *data = (float *)tensor->data;
            for (size_t i = 0; i < 10 && i < tensor->shape[0]; i++) {
                printf("Data[%zu]: %f\n", i, data[i]);
            }
        } else if (tensor->dtype == TYPE_INT8) {
            int8_t *data = (int8_t *)tensor->data;
            for (size_t i = 0; i < 10 && i < tensor->shape[0]; i++) {
                printf("Data[%zu]: %d\n", i, data[i]);
            }
        }

        // 释放内存
        TensorData_free(tensor);
    }
}