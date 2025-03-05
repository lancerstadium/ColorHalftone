#include "intp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OPR_LOG(...) fprintf(stdout, __VA_ARGS__)
#define OPR_ERR(...) fprintf(stderr, __VA_ARGS__)

// ======================================================================== //
//                               Functions
// ======================================================================== //

int Lut_pad_dict(char mode) {
    switch(mode) {
        case 's': return 1;
        case 'd': return 2;
        case 'y': return 2;
        case 'e': return 3;
        case 'h': return 3;
        case 'o': return 3;
        default : return 1;
    }
}

void Disp_u8_hwc(uint8_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", H, W, C);
    for (int h = 0; h < H; h++) {
        OPR_LOG(" [");
        for (int w = 0; w < W; w++) {
            if(w == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int c = 0; c < C; c++) {
                if(c == C - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(w == W - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_u32_hwc(uint32_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", H, W, C);
    for (int h = 0; h < H; h++) {
        OPR_LOG(" [");
        for (int w = 0; w < W; w++) {
            if(w == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int c = 0; c < C; c++) {
                if(c == C - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(w == W - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_s32_hwc(int32_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", H, W, C);
    for (int h = 0; h < H; h++) {
        OPR_LOG(" [");
        for (int w = 0; w < W; w++) {
            if(w == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int c = 0; c < C; c++) {
                if(c == C - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(w == W - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}


void Disp_u8_chw(uint8_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_s8_chw(int8_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_u32_chw(uint32_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_s32_chw(int32_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_s16_chw(int16_t *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%d", x[h * W * C + w * C + c]);
                } else {
                    OPR_LOG("%d,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            } else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

void Disp_f32_chw(float *x, int H, int W, int C) {
    // tensor type print
    OPR_LOG("Tensor(%d, %d, %d) = [\n", C, H, W);
    for (int c = 0; c < C; c++) {
        OPR_LOG(" [");
        for (int h = 0; h < H; h++) {
            if(h == 0) {
                OPR_LOG("[");
            } else {
                OPR_LOG("  [");
            }
            for (int w = 0; w < W; w++) {
                if(w == W - 1) {
                    OPR_LOG("%f", x[h * W * C + w * C + c]);
                }
                else {
                    OPR_LOG("%f,", x[h * W * C + w * C + c]);
                }
            }
            if(h == H - 1) {
                OPR_LOG("]");
            }
            else {
                OPR_LOG("],\n");
            }
        }
        OPR_LOG("],\n");
    }
    OPR_LOG("]\n");
}

// ======================================================================== //
//                               Operators
// ======================================================================== //

int32_t* DAlc_s32(int N) {
    int32_t* O = (int32_t*)malloc(N * sizeof(int32_t));
    if(O == NULL) {
        OPR_ERR("Alloc memory failed!\n");
        exit(EXIT_FAILURE);
    }
    return O;
}

uint8_t* DAlc_u8(int N) {
    uint8_t* O = (uint8_t*)malloc(N * sizeof(uint8_t));
    if(O == NULL) {
        OPR_ERR("Alloc memory failed!\n");
        exit(EXIT_FAILURE);
    }
    return O;
}

int8_t* DAlc_s8(int N) {
    int8_t* O = (int8_t*)malloc(N * sizeof(int8_t));
    if(O == NULL) {
        OPR_ERR("Alloc memory failed!\n");
        exit(EXIT_FAILURE);
    }
    return O;
}


void Copy_s32(int32_t* O, int32_t* I, int N) {
    // O [N] <- I [N]
    memcpy(O, I, N * sizeof(int32_t));
}

void Copy_u8(uint8_t* O, uint8_t* I, int N) {
    // O [N] <- I [N]
    memcpy(O, I, N * sizeof(uint8_t));
}

void DRlz_s32(int32_t* I) {
    if(I == NULL) {
        return;
    }
    free(I);
    I = NULL;
}

void DRlz_u8(uint8_t* I) {
    if(I == NULL) {
        return;
    }
    free(I);
    I = NULL;
}

void DRlz_s8(int8_t* I) {
    if(I == NULL) {
        return;
    }
    free(I);
    I = NULL;
}

void Zero_s32(int32_t* O, int N) {
    // O [N] <- 0
    memset(O, 0, N * sizeof(int32_t));
}

void Zero_u8(uint8_t* O, int N) {
    // O [N] <- 0
    memset(O, 0, N * sizeof(uint8_t));
}

void Zero_s8(int8_t* O, int N) {
    // O [N] <- 0
    memset(O, 0, N * sizeof(int8_t));
}

void Fill_s32(int32_t* O, int N, int32_t val) {
    // O [N] <- val
    for (int i = 0; i < N; i++) {
        O[i] = val;
    }
}

void Fill_u8(uint8_t* O, int N, uint8_t val) {
    // O [N] <- val
    for (int i = 0; i < N; i++) {
        O[i] = val;
    }
}

float inline Round_f32(float x) {
    // 3.6 -> 4.0
    // 3.4 -> 3.0
    return (x > 0) ? (x + 0.5) : (x - 0.5);
}

int32_t inline Round_s32_f32(float x) {
    return (x > 0) ? (int32_t)(x + 0.5) : (int32_t)(x - 0.5);
}

int32_t inline Round_Div_s32(int32_t x, int32_t y) {
    return (x > 0) ? (x + y / 2) / y : (x - y / 2) / y;
}

int16_t inline Round_Div_s16(int16_t x, int16_t y) {
    return (x > 0) ? (x + y / 2) / y : (x - y / 2) / y;
}

int8_t inline Round_Div_s8(int8_t x, int8_t y) {
    return (x > 0) ? (x + y / 2) / y : (x - y / 2) / y;
}

uint8_t inline Clamp_u8_s32(int32_t x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return x;
}

int32_t inline Clamp_s32_s32(int32_t x, int bitwidth, int sign) {
    if(sign) {
        if(x < -(1 << (bitwidth - 1))) {
            return -(1 << (bitwidth - 1));
        } else if(x > (1 << (bitwidth - 1)) - 1) {
            return (1 << (bitwidth - 1)) - 1;
        } else {
            return x;
        }
    } else {
        if(x < 0) {
            return 0;
        } else if(x > (1 << bitwidth) - 1) {
            return (1 << bitwidth) - 1;
        } else {
            return x;
        }
    }
}

int8_t inline Clamp_s8_s8(int8_t x, int bitwidth, int sign) {
    if(sign) { 
        if(x < -(1 << (bitwidth - 1))) {
            return -(1 << (bitwidth - 1));
        } else if(x > (1 << (bitwidth - 1)) - 1) {
            return (1 << (bitwidth - 1)) - 1;
        } else {
            return x;
        }
    } else {
        if(x < 0) {
            return 0;
        } else if(x > (1 << bitwidth) - 1) {
            return (1 << bitwidth) - 1;
        } else {
            return x;
        }
    }
}

int16_t inline Clamp_s16_s16(int16_t x, int bitwidth, int sign) {
    if(sign) { 
        if(x < -(1 << (bitwidth - 1))) {
            return -(1 << (bitwidth - 1));
        } else if(x > (1 << (bitwidth - 1)) - 1) {
            return (1 << (bitwidth - 1)) - 1;
        } else {
            return x;
        }
    } else {
        if(x < 0) {
            return 0;
        } else if(x > (1 << bitwidth) - 1) {
            return (1 << bitwidth) - 1;
        } else {
            return x;
        }
    }
}

void Mul_s32(int32_t* O, int32_t* I1, int32_t* I2, int N) {
    // O [N] <- I1 [N] * I2 [N]
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] * I2[i];
    }
}

void Mul_u8(uint8_t* O, uint8_t* I1, uint8_t* I2, int N) {
    // O [N] <- I1 [N] * I2 [N]
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] * I2[i];
    }
}


void MulQ_tile_f32_s32_hwc(int32_t* O, int32_t* I1, float* I2, int C, int H, int W, int sz) {
    // I1 [H, W, C]
    // I2 [sz, sz]
    int idx;
    int h, w, c, i;
    for(h = 0; h < H / sz; h++) {
        for(w = 0; w < W / sz; w++) {
            for(c = 0; c < C; c++) {
                for(i = 0; i < sz * sz; i++) {
                    idx = (h * sz + i / sz) * W * C + (w * sz + i % sz) * C + c;
                    O[idx] = Round_s32_f32(I1[idx] * I2[i]);
                }
            }
        }
    }
}

void MulQ_tile_s8_s32_hwc(int32_t* O, int32_t* I1, int8_t* I2, int scale, int offset, int C, int H, int W, int sz) {
    // I1 [H, W, C]
    // I2 [sz, sz]
    int idx;
    int h, w, c, i;
    for(h = 0; h < H / sz; h++) {
        for(w = 0; w < W / sz; w++) {
            for(c = 0; c < C; c++) {
                for(i = 0; i < sz * sz; i++) {
                    idx = (h * sz + i / sz) * W * C + (w * sz + i % sz) * C + c;
                    O[idx] = Round_Div_s32(I1[idx] * (I2[i] - offset), scale);
                    // if(c == 0 && w == 0 && h == 0) {
                    //     OPR_LOG("%d = %d * (%d - %d) / %d\n", O[idx], I1[idx], I2[i], offset, scale);
                    // }
                }
            }
        }
    }
}

void MulQ_tile_s16_s32_hwc(int32_t* O, int32_t* I1, int16_t* I2, int scale, int offset, int C, int H, int W, int sz) {
    // I1 [H, W, C]
    // I2 [sz, sz]
    int idx;
    int h, w, c, i;
    for(h = 0; h < H / sz; h++) {
        for(w = 0; w < W / sz; w++) {
            for(c = 0; c < C; c++) {
                for(i = 0; i < sz * sz; i++) {
                    idx = (h * sz + i / sz) * W * C + (w * sz + i % sz) * C + c;
                    O[idx] = Round_Div_s32(I1[idx] * (I2[i] - offset), scale);
                }
            }
        }
    }
}

void MulQ_tile_s16_s8_hwc(int8_t* O, int8_t* I1, int16_t* I2, int scale, int offset, int C, int H, int W, int sz) {
    // I1 [H, W, C]
    // I2 [sz, sz]
    int idx;
    int h, w, c, i;
    for(h = 0; h < H / sz; h++) {
        for(w = 0; w < W / sz; w++) {
            for(c = 0; c < C; c++) {
                for(i = 0; i < sz * sz; i++) {
                    idx = (h * sz + i / sz) * W * C + (w * sz + i % sz) * C + c;
                    O[idx] = Round_Div_s32(I1[idx] * (I2[i] - offset), scale);
                }
            }
        }
    }
}

void Add_s32(int32_t* O, int32_t* I1, int32_t* I2, int N) {
    // O [N] <- I1 [N] + I2 [N]
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] + I2[i];
    }
}

void Add_u8(uint8_t* O, uint8_t* I1, uint8_t* I2, int N) {
    // O [N] <- I1 [N] + I2 [N]
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] + I2[i];
    }
}

void Add_s8(int8_t* O, int8_t* I1, int8_t* I2, int N) {
    // O [N] <- I1 [N] + I2 [N]
    // use int32_t to avoid overflow
    int i;
    for(i = 0; i < N; i++) {
        O[i] = Clamp_s32_s32((int32_t)I1[i] + I2[i], 8, 1);
    }
}

void Addi_s32(int32_t* O, int32_t* I1, int32_t I2, int N) {
    // O [N] <- I1 [N] + I2
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] + I2;
    }
}

void Addi_u8(uint8_t* O, uint8_t* I1, uint8_t I2, int N) {
    // O [N] <- I1 [N] + I2
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I1[i] + I2;
    }
}

void Clamp_s32(int32_t* O, int32_t* I, int N, int bitwidth, int sign) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = Clamp_s32_s32(I[i], bitwidth, sign);
    }
}

void Clamp_s8(int8_t* O, int8_t* I, int N, int bitwidth, int sign) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = Clamp_s8_s8(I[i], bitwidth, sign);
    }
}

void Quant_s8_f32(int8_t* O, float* I, int N, int* scale, int* offset) {
    // 1. Min-Max: find scale factor and offset
    float min = I[0], max = I[0];
    int i;
    for(i = 1; i < N; i++) {
        if(I[i] < min) min = I[i];
        if(I[i] > max) max = I[i];
    }
    float scale_f = 127.0 / (max - min);
    *scale = (int)scale_f;
    *offset = (int)(-min * scale_f);
    // 2. Quantize
    for(i = 0; i < N; i++) {
        O[i] = (Clamp_s32_s32((int)(Round_f32(I[i] * scale_f) + *offset), 8, 1));
    }
}

void Quant_s16_f32(int16_t* O, float* I, int N, int* scale, int* offset) {
    // 1. Min-Max: find scale factor and offset
    float min = I[0], max = I[0];
    int i;
    for(i = 1; i < N; i++) {
        if(I[i] < min) min = I[i];
        if(I[i] > max) max = I[i];
    }
    float scale_f = 32767.0 / (max - min);
    *scale = (int)scale_f;
    *offset = (int)(-min * scale_f);
    // 2. Quantize
    for(i = 0; i < N; i++) {
        O[i] = (Clamp_s32_s32((int)(Round_f32(I[i] * scale_f) + *offset), 16, 1));
    }
}

void Dequant_s8_f32(float* O, int8_t* I, int N, int scale, int offset) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = (float)(I[i] - offset) / scale;
    }
}

void Dequant_s16_f32(float* O, int16_t* I, int N, int scale, int offset) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = (float)(I[i] - offset) / scale;
    }
}

void Cast_s32_u8(int32_t* O, uint8_t* I, int N) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = I[i];
    }
}

void Cast_s8_u8(int8_t* O, uint8_t* I, int N) {
    // O [N] <- I [N] - 128
    int i;
    for(i = 0; i < N; i++) {
        O[i] = (int)I[i] - 128;
    }
}

void Cast_u8_s32(uint8_t* O, int32_t* I, int N) {
    int i;
    for(i = 0; i < N; i++) {
        O[i] = Clamp_u8_s32(I[i]);
    }
}

void Cast_u8_s8(uint8_t* O, int8_t* I, int N) {
    // O [N] <- I [N] + 128
    int i;
    for(i = 0; i < N; i++) {
        O[i] = (int)I[i] + 128;
    }
}


void Pad_u8_hwc(uint8_t* O, uint8_t* I, int C, int H, int W, int pad) {
    int i, j, k;
    for(i = 0; i < H; i++) {
        for(j = 0; j < W; j++) {
            for(k = 0; k < C; k++) {
                O[(i * (W + pad) + j) * C + k] = I[(i * W + j) * C + k];
            }
        }
    }
    for(i = H; i < H + pad; i++) {
        for(j = 0; j < W; j++) {
            for(k = 0; k < C; k++) {
                O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + j) * C + k];
            }
        }
    }
    for(i = 0; i < H; i++) {
        for(j = W; j < W + pad; j++) {
            for(k = 0; k < C; k++) {
                O[(i * (W + pad) + j) * C + k] = I[(i * W + (W - 1)) * C + k];
            }
        }
    }
    for(i = H; i < H + pad; i++) {
        for(j = W; j < W + pad; j++) {
            for(k = 0; k < C; k++) {
                O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + (W - 1)) * C + k];
            }
        }
    }
}



void Pad_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int pad, int rb, int reflect) {
    int i, j, k;
    if(rb) {    // if right-bottom
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * (W + pad) + j) * C + k] = I[(i * W + j) * C + k];
                }
            }
        }
        for(i = H; i < H + pad; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(((H - 1) - (i - H)) * W + j) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + j) * C + k];
                    }
                }
            }
        }
        for(i = 0; i < H; i++) {
            for(j = W; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(i * W + (W - 1) - (j - W)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(i * W + (W - 1)) * C + k];
                    }
                }
            }
        }
        for(i = H; i < H + pad; i++) {
            for(j = W; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(((H - 1) - (i - H)) * W + (W - 1) - (j - W)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + (W - 1)) * C + k];
                    }
                }
            }
        }
    } else {    // if left-top
        for(i = pad; i < H + pad; i++) {
            for(j = pad; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + (j - pad)) * C + k];
                }
            }
        }
        for(i = 0; i < pad; i++) {
            for(j = 0; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((2 * pad - 2 - i) * W + (j - pad)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(0 * W + (j - pad)) * C + k];
                    }
                }
            }
        }
        for(i = pad; i < H + pad; i++) {
            for(j = 0; j < pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + (2 * pad - 2 - j)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + 0) * C + k];
                    }
                }
            }
        }
        for(i = 0; i < pad; i++) {
            for(j = 0; j < pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((2 * pad - 2 - i) * W + (2 * pad - 2 - j)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(0 * W + 0) * C + k];
                    }
                }
            }
        }

    }
}

void Pad_s8_hwc(int8_t* O, int8_t* I, int C, int H, int W, int pad, int rb, int reflect) {
    int i, j, k;
    if(rb) {    // if right-bottom
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * (W + pad) + j) * C + k] = I[(i * W + j) * C + k];
                }
            }
        }
        for(i = H; i < H + pad; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(((H - 1) - (i - H)) * W + j) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + j) * C + k];
                    }
                }
            }
        }
        for(i = 0; i < H; i++) {
            for(j = W; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(i * W + (W - 1) - (j - W)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(i * W + (W - 1)) * C + k];
                    }
                }
            }
        }
        for(i = H; i < H + pad; i++) {
            for(j = W; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[(((H - 1) - (i - H)) * W + (W - 1) - (j - W)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((H - 1) * W + (W - 1)) * C + k];
                    }
                }
            }
        }
    } else {    // if left-top
        for(i = pad; i < H + pad; i++) {
            for(j = pad; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + (j - pad)) * C + k];
                }
            }
        }
        for(i = 0; i < pad; i++) {
            for(j = 0; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((2 * pad - 2 - i) * W + (j - pad)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(0 * W + (j - pad)) * C + k];
                    }
                }
            }
        }
        for(i = pad; i < H + pad; i++) {
            for(j = 0; j < pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + (2 * pad - 2 - j)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[((i - pad) * W + 0) * C + k];
                    }
                }
            }
        }
        for(i = 0; i < pad; i++) {
            for(j = 0; j < pad; j++) {
                for(k = 0; k < C; k++) {
                    if(reflect) {
                        O[(i * (W + pad) + j) * C + k] = I[((2 * pad - 2 - i) * W + (2 * pad - 2 - j)) * C + k];
                    } else {
                        O[(i * (W + pad) + j) * C + k] = I[(0 * W + 0) * C + k];
                    }
                }
            }
        }
    }
}

void Crop_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int pad, int rb) {
    int i, j, k;
    int h = H - pad;
    int w = W - pad;

    if(rb == 0) {    // if right-bottom
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * w + j) * C + k] = I[(i * (W + pad) + j) * C + k];
                }
            }
        }
    } else {    // if left-top
        for(i = pad; i < H + pad; i++) {
            for(j = pad; j < W + pad; j++) {
                for(k = 0; k < C; k++) {
                    O[((i - pad) * w + (j - pad)) * C + k] = I[(i * (W + pad) + j) * C + k];
                }
            }
        }
    }
}

// img<hwc> ==rot90 * times==> img<hwc>
void Rot_u8_hwc(uint8_t* O, uint8_t* I, int C, int H, int W, int times) {
    int i, j, k;
    int h = H;
    int w = W;
    int time = times % 4;

    if(time == 0) {
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * W + j) * C + k] = I[(i * W + j) * C + k];
                }
            }
        }
    } else if(time == 1) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[(j * h + (h - i - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 2) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((h - i - 1) * w + (w - j - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 3) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((w - j - 1) * h + i) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else {
        OPR_ERR("Invalid times: %d\n", times);
        exit(EXIT_FAILURE);
    }
}

void Rot_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int times) {
    int i, j, k;
    int h = H;
    int w = W;
    int time = times % 4;

    if(time == 0) {
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * W + j) * C + k] = I[(i * W + j) * C + k];
                }
            }
        }
    } else if(time == 1) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[(j * h + (h - i - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 2) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((h - i - 1) * w + (w - j - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 3) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((w - j - 1) * h + i) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else {
        OPR_ERR("Invalid times: %d\n", times);
        exit(EXIT_FAILURE);
    }
}

void Rot_s8_hwc(int8_t* O, int8_t* I, int C, int H, int W, int times) {
    int i, j, k;
    int h = H;
    int w = W;
    int time = times % 4;

    if(time == 0) {
        for(i = 0; i < H; i++) {
            for(j = 0; j < W; j++) {
                for(k = 0; k < C; k++) {
                    O[(i * W + j) * C + k] = I[(i * W + j) * C + k];
                }
            }
        }
    } else if(time == 1) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[(j * h + (h - i - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 2) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((h - i - 1) * w + (w - j - 1)) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else if(time == 3) {
        for(i = 0; i < h; i++) {
            for(j = 0; j < w; j++) {
                for(k = 0; k < C; k++) {
                        O[((w - j - 1) * h + i) * C + k] = I[(i * w + j) * C + k];
                }
            }
        }
    } else {
        OPR_ERR("Invalid times: %d\n", times);
        exit(EXIT_FAILURE);
    }
}

void Seg_s32_hwc(int32_t* O1, int32_t* O2, int32_t* I, int N, int interval) {
    int i;
    if(interval <= 0) { interval = 1; }
    else if(interval > 31) { interval = 31; }
    int mask = (1 << interval);
    for(i = 0; i < N; i++) {
        if(O1) { O1[i] = I[i] % mask; }
        if(O2) { O2[i] = I[i] / mask; }
    }
}

void Seg_u8_hwc(uint8_t* O1, uint8_t* O2, uint8_t* I, int N, int interval) {
    int i;
    if(interval <= 0) { interval = 1; }
    else if(interval > 7) { interval = 7; }
    int mask = (1 << interval);
    for(i = 0; i < N; i++) {
        if(O1) { O1[i] = I[i] % mask; }
        if(O2) { O2[i] = I[i] / mask; }
    }
}
void Seg_s8_hwc(int8_t* O1, int8_t* O2, int8_t* I, int N, int interval) {
    int i;
    if(interval <= 0) { interval = 1; }
    else if(interval > 7) { interval = 7; }
    int mask = (1 << interval);
    for(i = 0; i < N; i++) {
        if(O1) { O1[i] = I[i] % mask; }
        if(O2) { O2[i] = I[i] / mask; }
    }
}


#include <math.h>

void Intp_bicubic_s32_hwc(
    int32_t* O, 
    int32_t* I, 
    int C, 
    int H, 
    int W, 
    int upscale
) { 
    int newH = H * upscale; 
    int newW = W * upscale; 
    const float a = -0.75f;

    for (int oy = 0; oy < newH; oy++) {
        for (int ox = 0; ox < newW; ox++) {
            // 计算对应的输入坐标
            float ix = ((float)ox + 0.5f) / upscale - 0.5f;
            float iy = ((float)oy + 0.5f) / upscale - 0.5f;
    
            int x0 = (int)floorf(ix);
            int y0 = (int)floorf(iy);
    
            float dx = ix - x0;
            float dy = iy - y0;
    
            float sum[C];
            for (int c = 0; c < C; c++) {
                sum[c] = 0.0f;
            }
    
            for (int j = -1; j <= 2; j++) {
                int y = y0 + j;
                y = y < 0 ? 0 : (y >= H ? H-1 : y);
    
                float ty = dy - j;
                float abs_ty = fabsf(ty);
                float wy;
                if (abs_ty <= 1.0f) {
                    wy = (a + 2.0f) * abs_ty * abs_ty * abs_ty - (a + 3.0f) * abs_ty * abs_ty + 1.0f;
                } else if (abs_ty < 2.0f) {
                    wy = a * abs_ty * abs_ty * abs_ty - 5.0f * a * abs_ty * abs_ty + 8.0f * a * abs_ty - 4.0f * a;
                } else {
                    wy = 0.0f;
                }
    
                for (int i = -1; i <= 2; i++) {
                    int x = x0 + i;
                    x = x < 0 ? 0 : (x >= W ? W-1 : x);
    
                    float tx = dx - i;
                    float abs_tx = fabsf(tx);
                    float wx;
                    if (abs_tx <= 1.0f) {
                        wx = (a + 2.0f) * abs_tx * abs_tx * abs_tx - (a + 3.0f) * abs_tx * abs_tx + 1.0f;
                    } else if (abs_tx < 2.0f) {
                        wx = a * abs_tx * abs_tx * abs_tx - 5.0f * a * abs_tx * abs_tx + 8.0f * a * abs_tx - 4.0f * a;
                    } else {
                        wx = 0.0f;
                    }
    
                    float weight = wx * wy;
    
                    // 获取输入像素的地址
                    int32_t* pixel_ptr = I + (y * W + x) * C;
                    for (int c = 0; c < C; c++) {
                        sum[c] += pixel_ptr[c] * weight;
                    }
                }
            }
    
            // 处理sum，写入O
            int32_t* out_ptr = O + (oy * newW + ox) * C;
            for (int c = 0; c < C; c++) {
                float val = sum[c];
                val = roundf(val);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                out_ptr[c] = (int32_t)val;
            }
        }
    }
}


    // https://blog.csdn.net/MR_kdcon/article/details/123661418
void Intp_simplex4d_s32_hwc(
    int32_t* O,     //    
    int32_t* I,     //    0 ~ 255
    int8_t* LUT,    // -127 ~ 127
    int C,
    int H,
    int W,          
    int bitwidth,   // 8
    int interval,   // 4
    int upscale,    // 4
    int norm,       // 4
    int bias,       // 0 | 127
    char mode       // 's'|'d'|'y'
) {
    int pad = Lut_pad_dict(mode);
    int h = H - pad;
    int w = W - pad;
    int q = 1 << interval;
    int L = (1 << (bitwidth - interval)) + 1;
    int S = upscale * upscale;

    int i, j, k, x, y;

    OPR_LOG("mode = %c, q = %d, L = %d, h = %d, w = %d\n", mode, q, L, h, w);

    uint8_t LSB [C * h * w * 4];
    uint8_t MSB [C * h * w * 4];
    int8_t* p   [16];
    int8_t* v   [3];
    int8_t  u   [4];
   
    // MSB & LSB
    switch(mode) {
        case 's': {
            for(j = 0; j < h; j++) {
                for(k = 0; k < w; k++) {
                    for(i = 0; i < C; i++) {
                        // OPR_LOG("(%d, %d, %d): %d/%d, %d/%d\n", j, k, i, i * h * w * 4 + j * w * 4 + k * 4 + 3, C * h * w * 4, (j + 1) * W * C + (k + 1) * C + i, H * W * C);
                        // OPR_LOG("%d, %d\n", i * h * w * 4 + (j + 0) * w * 4 + (k + 0) * 4 + 0, (j + 0) * W * C + (k + 0) * C + i);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 0) * W * C + (k + 1) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 0) * W * C + (k + 1) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 1) * W * C + (k + 0) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 1) * W * C + (k + 0) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 1) * W * C + (k + 1) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 1) * W * C + (k + 1) * C + i] & (q - 1);
                    }
                }
            }
            break;
        }
        case 'd': {
            for(j = 0; j < h; j++) {
                for(k = 0; k < w; k++) {
                    for(i = 0; i < C; i++) {
                        // OPR_LOG("(%d, %d, %d): %d/%d, %d/%d\n", j, k, i, i * h * w * 4 + j * w * 4 + k * 4 + 3, C * h * w * 4, (j + 2) * W * C + (k + 2) * C + i, H * W * C);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 0) * W * C + (k + 2) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 0) * W * C + (k + 2) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 2) * W * C + (k + 0) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 2) * W * C + (k + 0) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 2) * W * C + (k + 2) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 2) * W * C + (k + 2) * C + i] & (q - 1);
                    }
                }
            }
            break;
        }
        case 'y': { 
            
            for(j = 0; j < h; j++) {
                for(k = 0; k < w; k++) {
                    for(i = 0; i < C; i++) {
                        // OPR_LOG("(%d, %d, %d): %d/%d, %d/%d\n", j, k, i, i * h * w * 4 + j * w * 4 + k * 4 + 3, C * h * w * 4, (j + 2) * W * C + (k + 1) * C + i, H * W * C);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 0] = (uint8_t)I[(j + 0) * W * C + (k + 0) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 1) * W * C + (k + 1) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 1] = (uint8_t)I[(j + 1) * W * C + (k + 1) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 1) * W * C + (k + 2) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 2] = (uint8_t)I[(j + 1) * W * C + (k + 2) * C + i] & (q - 1);
                        MSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 2) * W * C + (k + 1) * C + i] >> interval;
                        LSB[i * h * w * 4 + j * w * 4 + k * 4 + 3] = (uint8_t)I[(j + 2) * W * C + (k + 1) * C + i] & (q - 1);
                    }
                }
            }
            break;
        }
        default: {
            OPR_ERR("Mode `%c` is not supported.\n", mode);
            exit(EXIT_FAILURE);
        }
    }

    // 4D Interpolation
    for(j = 0; j < h; j++) {
        for(k = 0; k < w; k++) {
            for(i = 0; i < C; i++) {
                uint8_t* m = &MSB[i * h * w * 4 + (j + 0) * w * 4 + (k + 0) * 4 + 0];
                uint8_t* l = &LSB[i * h * w * 4 + (j + 0) * w * 4 + (k + 0) * 4 + 0];
                // OPR_LOG("(%d/%d, %d/%d, %d/%d)\n", j, h, k, w, i, C);
                p[0b0000] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 0) * L * S + (m[3] + 0) * S];
                p[0b0001] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 0) * L * S + (m[3] + 1) * S];
                p[0b0010] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 1) * L * S + (m[3] + 0) * S];
                p[0b0011] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 1) * L * S + (m[3] + 1) * S];
                p[0b0100] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 0) * L * S + (m[3] + 0) * S];
                p[0b0101] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 0) * L * S + (m[3] + 1) * S];
                p[0b0110] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 1) * L * S + (m[3] + 0) * S];
                p[0b0111] = &LUT[(m[0] + 0) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 1) * L * S + (m[3] + 1) * S];
                p[0b1000] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 0) * L * S + (m[3] + 0) * S];
                p[0b1001] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 0) * L * S + (m[3] + 1) * S];
                p[0b1010] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 1) * L * S + (m[3] + 0) * S];
                p[0b1011] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 0) * L * L * S + (m[2] + 1) * L * S + (m[3] + 1) * S];
                p[0b1100] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 0) * L * S + (m[3] + 0) * S];
                p[0b1101] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 0) * L * S + (m[3] + 1) * S];
                p[0b1110] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 1) * L * S + (m[3] + 0) * S];
                p[0b1111] = &LUT[(m[0] + 1) * L * L * L * S + (m[1] + 1) * L * L * S + (m[2] + 1) * L * S + (m[3] + 1) * S];

                if(l[0] > l[1]) {
                    if(l[0] > l[2]) {
                        if(l[0] > l[3]) {
                            u[0] = l[0]; u[1] = l[3]; u[2] = l[2]; u[3] = l[1];
                            v[0] = p[0b1000]; v[1] = p[0b1001]; v[2] = p[0b1011];
                        } else if(l[1] > l[3]) {
                            u[0] = l[0]; u[1] = l[2]; u[2] = l[1]; u[3] = l[3];
                            v[0] = p[0b1000]; v[1] = p[0b1010]; v[2] = p[0b1110];
                        } else if(l[2] > l[3]) {
                            u[0] = l[0]; u[1] = l[2]; u[2] = l[3]; u[3] = l[1];
                            v[0] = p[0b1000]; v[1] = p[0b1001]; v[2] = p[0b1011];
                        } else {
                            u[0] = l[3]; u[1] = l[0]; u[2] = l[2]; u[3] = l[1];
                            v[0] = p[0b0001]; v[1] = p[0b1001]; v[2] = p[0b1011];
                        }
                    } else if(l[1] > l[2]) {
                        if(l[0] > l[3]) {
                            u[0] = l[0]; u[1] = l[3]; u[2] = l[1]; u[3] = l[2];
                            v[0] = p[0b1000]; v[1] = p[0b1001]; v[2] = p[0b1101];
                        } else if(l[1] > l[3]) {
                            u[0] = l[0]; u[1] = l[1]; u[2] = l[3]; u[3] = l[2];
                            v[0] = p[0b1000]; v[1] = p[0b1100]; v[2] = p[0b1101];
                        } else if(l[2] > l[3]) {
                            u[0] = l[0]; u[1] = l[1]; u[2] = l[2]; u[3] = l[3];
                            v[0] = p[0b1000]; v[1] = p[0b1100]; v[2] = p[0b1110];
                        } else {
                            u[0] = l[3]; u[1] = l[0]; u[2] = l[1]; u[3] = l[2];
                            v[0] = p[0b0001]; v[1] = p[0b1001]; v[2] = p[0b1101];
                        }
                    } else {
                        if(l[0] > l[3]) {
                            u[0] = l[2]; u[1] = l[3]; u[2] = l[0]; u[3] = l[1];
                            v[0] = p[0b0010]; v[1] = p[0b0011]; v[2] = p[0b1011];
                        } else if(l[1] > l[3]) {
                            u[0] = l[2]; u[1] = l[1]; u[2] = l[0]; u[3] = l[3];
                            v[0] = p[0b0010]; v[1] = p[0b1010]; v[2] = p[0b1110];
                        } else if(l[2] > l[3]) {
                            u[0] = l[2]; u[1] = l[0]; u[2] = l[3]; u[3] = l[1];
                            v[0] = p[0b0010]; v[1] = p[0b1010]; v[2] = p[0b1011];
                        } else {
                            u[0] = l[3]; u[1] = l[2]; u[2] = l[0]; u[3] = l[1];
                            v[0] = p[0b0001]; v[1] = p[0b0011]; v[2] = p[0b1011];
                        }
                    }
                } else {
                    if(l[0] > l[2]) {
                        if(l[0] > l[3]) {
                            u[0] = l[1]; u[1] = l[0]; u[2] = l[3]; u[3] = l[2];
                            v[0] = p[0b0100]; v[1] = p[0b1100]; v[2] = p[0b1101];
                        } else if(l[1] > l[3]) {
                            u[0] = l[1]; u[1] = l[3]; u[2] = l[0]; u[3] = l[2];
                            v[0] = p[0b0100]; v[1] = p[0b0101]; v[2] = p[0b1101];
                        } else if(l[2] > l[3]) {
                            u[0] = l[1]; u[1] = l[0]; u[2] = l[2]; u[3] = l[3];
                            v[0] = p[0b0100]; v[1] = p[0b1100]; v[2] = p[0b1110];
                        } else {
                            u[0] = l[3]; u[1] = l[1]; u[2] = l[0]; u[3] = l[2];
                            v[0] = p[0b0001]; v[1] = p[0b0101]; v[2] = p[0b1101];
                        }
                    } else if(l[1] > l[2]) {
                        if(l[0] > l[3]) {
                            u[0] = l[1]; u[1] = l[2]; u[2] = l[0]; u[3] = l[3];
                            v[0] = p[0b0100]; v[1] = p[0b0110]; v[2] = p[0b1110];
                        } else if(l[1] > l[3]) {
                            u[0] = l[1]; u[1] = l[3]; u[2] = l[2]; u[3] = l[0];
                            v[0] = p[0b0100]; v[1] = p[0b0101]; v[2] = p[0b0111];
                        } else if(l[2] > l[3]) {
                            u[0] = l[1]; u[1] = l[2]; u[2] = l[3]; u[3] = l[0];
                            v[0] = p[0b0100]; v[1] = p[0b0110]; v[2] = p[0b0111];
                        } else {
                            u[0] = l[3]; u[1] = l[1]; u[2] = l[2]; u[3] = l[0];
                            v[0] = p[0b0001]; v[1] = p[0b0101]; v[2] = p[0b0111];
                        }
                    } else {
                        if(l[0] > l[3]) {
                            u[0] = l[2]; u[1] = l[1]; u[2] = l[0]; u[3] = l[3];
                            v[0] = p[0b0010]; v[1] = p[0b0110]; v[2] = p[0b1110];
                        } else if(l[1] > l[3]) {
                            u[0] = l[2]; u[1] = l[1]; u[2] = l[3]; u[3] = l[0];
                            v[0] = p[0b0010]; v[1] = p[0b0011]; v[2] = p[0b0111];
                        } else if(l[2] > l[3]) {
                            u[0] = l[2]; u[1] = l[3]; u[2] = l[1]; u[3] = l[0];
                            v[0] = p[0b0010]; v[1] = p[0b0011]; v[2] = p[0b0111];
                        } else {
                            u[0] = l[3]; u[1] = l[2]; u[2] = l[1]; u[3] = l[0];
                            v[0] = p[0b0001]; v[1] = p[0b0011]; v[2] = p[0b0111];
                        }
                    }
                }
                // OPR_LOG("(%d/%d, %d/%d, %d/%d)\n", j, h, k, w, i, C);
                // Calculate
                for(y = 0; y < upscale; y++) {
                    for(x = 0; x < upscale; x++) {
                        int32_t val = Round_Div_s32(((int32_t)(q - u[0]) * (int32_t)p[0b0000][y * upscale + x] + (int32_t)(u[0] - u[1]) * (int32_t)v[0][y * upscale + x] + (int32_t)(u[1] - u[2]) * (int32_t)v[1][y * upscale + x] + (int32_t)(u[2] - u[3]) * (int32_t)v[2][y * upscale + x] + (int32_t)(u[3]) * (int32_t)p[0b1111][y * upscale + x]), q);
                        O[(j * upscale + y) * (upscale * w) * C + (k * upscale + x) * C + i] = val / norm + bias;
                    }
                }
            }
        }
    }
}

void Intp_depthwise_s32_hwc(
    int32_t* O,     // [H2, W2, C]
    int32_t* I,     // [HP, WP, C]
    int8_t* LUT,    // -127 ~ 127
    int C,
    int H,
    int W,          
    int ksz,        // 3
    int upscale,    // 4
    int dense,      // if dense
    int hl          // high(1) | low (0)
) {
    int pad = 2;
    int ofs = 0;
    int h = H - pad;
    int w = W - pad;
    int c = C * ksz * ksz;

    int i, j, k, m, n, l, p;
    int L;
    int idx;
    if(hl)  { L = 64; ofs = 32; }
    else    { L =  4; ofs =  0; }

    int32_t xlist[h * w * c];

    // OPR_LOG("h = %d, w = %d, c = %d\n", h, w, c);

    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            for(k = 0; k < C; k++) {
                idx = 0;
                for(m = 0; m < ksz; m++) {
                    for(n = 0; n < ksz; n++) {
                        xlist[(i * w + j) * c + k * ksz * ksz + (m * ksz + n)] = I[((i + m) * W + (j + n)) * C + k] + ofs + idx * L;
                        idx++;
                    }
                }
            }
        }
    }

    // Disp_s32_chw(I, H, W, C);
    // Disp_s32_hwc(xlist, h, w, c);
    // Disp_s8_chw(LUT, 36, 16, 1);
    int32_t val[upscale * upscale];
    int8_t* lut;
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            for(k = 0; k < C; k++) {
                if(dense) {
                    ofs = I[((i + pad) * W + (j + pad)) * C + k];
                } else {
                    ofs = 0;
                }
                for(l = 0; l < upscale * upscale; l++) {
                    val[l] = 0;
                }
                for(l = 0; l < ksz * ksz; l++) {
                    lut = &LUT[xlist[(i * w + j) * c + k * ksz * ksz + l] * upscale * upscale];
                    for(p = 0; p < upscale * upscale; p++) {
                        val[p] += lut[p];
                    }
                }
                for(l = 0; l < upscale * upscale; l++) {
                    // OPR_LOG("(%d, %d, %d, %d): %d\n", i, j, k, l, val[l]);
                    int32_t trg = Round_Div_s32(val[l], ksz * ksz) + ofs;
                    if(hl) {
                        trg = Clamp_s32_s32(trg, 6, 1);
                    } else {
                        trg = Clamp_s32_s32(trg, 2, 0);
                    }
                    O[(i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k] = trg;
                }
            }
        }
    }
}


void Intp_pointwise_s32_hwc(
    int32_t* O,     
    int32_t* I,     
    int8_t* LUT,    // -127 ~ 127
    int16_t* RAT,
    int C,
    int H,
    int W,          
    int scale,
    int offset,
    int upscale,    // 4
    int dense,      // if dense
    int clamp8,     // if clamp 8
    int hl          // high(1) | low (0)
) {

    int h = H / upscale;
    int w = W / upscale;
    int i, j, k, l, m, S = 0;
    int ofs, idx, idxx;
    int8_t* lut[upscale * upscale];

    int min, max;
    if(hl)  { min = -32; max = 31; }
    else    { min =   0; max =  3; }

    int min_[upscale * upscale];
    int max_[upscale * upscale];

    for(l = 0; l < upscale * upscale; l++) {
        lut[l] = &LUT[S * upscale * upscale];
        min_[l] = Round_Div_s32(min * (RAT[l] - offset), scale);
        max_[l] = Round_Div_s32(max * (RAT[l] - offset), scale);
        int cnt_ = max_[l] - min_[l] + 1;
        // printf("S: %d, min: %d, max: %d, cnt: %d\n", S, min_[l], max_[l], cnt_);
        ofs = cnt_;
        S += ofs;
    }

    int32_t val[upscale * upscale];
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            for(k = 0; k < C; k++) {
                for(m = 0; m < upscale * upscale; m++) {
                    val[m] = 0;
                }
                for(l = 0; l < upscale * upscale; l++) {
                    idx = (i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k;
                    idxx = (I[idx] - min_[l]);
                    // Accumulate
                    for(m = 0; m < upscale * upscale; m++) {
                        val[m] += lut[l][idxx * upscale * upscale + m];
                        // if(l == 0) {
                        //     printf("[%d](%d, %d, %d, %d, %d) idxx: %d, val: %d\n", idx, i, j, k, l, m, idxx, val[l]);
                        // }
                    }
                }
                for(l = 0; l < upscale * upscale; l++) {
                    idx = (i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k;
                    // Average
                    if(dense) {
                        ofs = I[idx];
                    } else {
                        ofs = 0;
                    }
                    int32_t trg = Round_Div_s32(val[l], upscale * upscale) + ofs;
                    if(hl && !clamp8) {
                        trg = Clamp_s32_s32(trg, 6, 1);
                    } else if (!hl && !clamp8) {
                        trg = Clamp_s32_s32(trg, 2, 0);
                    } else if(clamp8) {
                        trg = Clamp_s32_s32(trg, 8, 1);
                    }
                    O[(i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k] = trg;
                }
            }
        }
    }
}


void Intp_depthwise_s8_hwc (
    int8_t* O,      // [H2, W2, C]
    int8_t* I,      // [HP, WP, C] padded input
    int8_t* LUT,    // -127 ~ 127
    int C,
    int H,
    int W,          
    int ksz,        // 3
    int upscale,    // 4
    int dense,      // if dense
    int hl          // high(1)|low(0)
) {
    const int pad = 2;
    const int h = H - pad;
    const int w = W - pad;
    
    const int L = hl ? 64 : 4;
    const int init_ofs = hl ? 32 : 0;  

    int i, j, k, l, p;
    const int ksz_sq = ksz * ksz;
    const int us_sq = upscale * upscale;
    int32_t val[upscale * upscale];      

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            for (k = 0; k < C; k++) {
                const int dense_ofs = dense ? I[((i + pad) * W + (j + pad)) * C + k] : 0;
                
                for (l = 0; l < us_sq; l++) {
                    val[l] = 0;
                }
                
                for (l = 0; l < ksz_sq; l++) {
                    const int m = l / ksz;    
                    const int n = l % ksz;    

                    const int i_I = i + m;
                    const int j_I = j + n;
                    const int x_val = I[(i_I * W + j_I) * C + k] + init_ofs + l * L;
                    
                    const int8_t* lut = &LUT[x_val * us_sq];
                    for (p = 0; p < us_sq; p++) {
                        val[p] += lut[p];
                    }
                }

                for (l = 0; l < us_sq; l++) {
                    int32_t trg = Round_Div_s32(val[l], ksz_sq) + dense_ofs;

                    trg = hl ? Clamp_s32_s32(trg, 6, 1) : Clamp_s32_s32(trg, 2, 0);
                    
                    const int oi = i * upscale + l / upscale;
                    const int oj = j * upscale + l % upscale;
                    O[(oi * w * upscale + oj) * C + k] = trg;
                }
            }
        }
    }
}


void Intp_pointwise_s8_hwc(
    int8_t* O,     
    int8_t* I,     
    int8_t* LUT,    // -127 ~ 127
    int16_t* RAT,
    int C,
    int H,
    int W,          
    int scale,
    int offset,
    int upscale,    // 4
    int dense,      // if dense
    int clamp8,     // if clamp 8
    int hl          // high(1) | low (0)
) {

    int h = H / upscale;
    int w = W / upscale;
    int i, j, k, l, m, S = 0;
    int ofs, idx, idxx;
    int8_t* lut[upscale * upscale];

    int min, max;
    if(hl)  { min = -32; max = 31; }
    else    { min =   0; max =  3; }

    int min_[upscale * upscale];
    int max_[upscale * upscale];

    for(l = 0; l < upscale * upscale; l++) {
        lut[l] = &LUT[S * upscale * upscale];
        min_[l] = Round_Div_s32(min * (RAT[l] - offset), scale);
        max_[l] = Round_Div_s32(max * (RAT[l] - offset), scale);
        int cnt_ = max_[l] - min_[l] + 1;
        ofs = cnt_;
        S += ofs;
    }

    int32_t val[upscale * upscale];
    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            for(k = 0; k < C; k++) {
                for(m = 0; m < upscale * upscale; m++) {
                    val[m] = 0;
                }
                for(l = 0; l < upscale * upscale; l++) {
                    idx = (i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k;
                    idxx = (I[idx] - min_[l]);
                    // Accumulate
                    for(m = 0; m < upscale * upscale; m++) {
                        val[m] += lut[l][idxx * upscale * upscale + m];
                    }
                }
                for(l = 0; l < upscale * upscale; l++) {
                    idx = (i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k;
                    // Average
                    if(dense) {
                        ofs = I[idx];
                    } else {
                        ofs = 0;
                    }
                    int32_t trg = Round_Div_s32(val[l], upscale * upscale) + ofs;
                    if(hl && !clamp8) {
                        trg = Clamp_s32_s32(trg, 6, 1);
                    } else if (!hl && !clamp8) {
                        trg = Clamp_s32_s32(trg, 2, 0);
                    } else if(clamp8) {
                        trg = Clamp_s32_s32(trg, 8, 1);
                    }
                    O[(i * upscale + l / upscale) * (upscale * w) * C + (j * upscale + l % upscale) * C + k] = trg;
                }
            }
        }
    }
}




void Intp_fuse_s8_hwc(
    int8_t* O,      // 输出 [H2, W2, C]
    int8_t* I,      // 输入 [H, W, C] (MSB高6位 | LSB低2位)
    int8_t* LUT_LSB, 
    int8_t* LUT_MSB, // 双LUT
    int16_t* RAT_LSB,   // 量化参数
    int16_t* RAT_MSB,   // 量化参数
    int C,
    int H,
    int W,
    int scale_LSB,
    int offset_LSB,
    int scale_MSB,
    int offset_MSB,
    int upscale,
    int ksz,
    int dense,
    int clamp8,
    int dp
) {
    const int pad = 2;
    const int h = dp ? (H - pad) : (H / upscale);
    const int w = dp ? (W - pad) : (W / upscale);
    const int us_sq = upscale * upscale;
    const int ksz_sq = ksz * ksz;
    int32_t val_MSB[us_sq], val_LSB[us_sq];

    if(dp) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < C; k++) {
                    // depthwise 原始输入位置（考虑padding）
                    int dense_MSB = dense ? (I[((i + pad) * W + (j + pad)) * C + k] / 4) : 0;
                    int dense_LSB = dense ? (I[((i + pad) * W + (j + pad)) * C + k] % 4) : 0;
                    // 同时重置MSB和LSB的中间结果
                    for(int l = 0; l < us_sq; l++) {
                        val_MSB[l] = 0;
                        val_LSB[l] = 0;
                    }
                    // 3x3卷积核遍历
                    for (int m = 0; m < ksz; m++) {
                        for (int n = 0; n < ksz; n++) {
                            const int8_t val = I[((i + m) * W + (j + n)) * C + k];
                            // MSB通路参数
                            const int idx_MSB = val / 4 + 32 + (m*ksz + n)*64;    // 初始偏移32，L=64
                            const int8_t* lut_msb = &LUT_MSB[idx_MSB * us_sq];
                            // LSB通路参数
                            const int idx_LSB = val % 4 +  0 + (m*ksz + n)* 4;    // 初始偏移0，L=4
                            const int8_t* lut_lsb = &LUT_LSB[idx_LSB * us_sq];
                            // 累加插值结果
                            for (int p = 0; p < us_sq; p++) {
                                val_MSB[p] += lut_msb[p];
                                val_LSB[p] += lut_lsb[p];
                            }
                        }
                    }
                    // 结果合并
                    for (int p = 0; p < us_sq; p++) {
                        int32_t trg_MSB = Round_Div_s32(val_MSB[p], ksz_sq) + dense_MSB;
                        int32_t trg_LSB = Round_Div_s32(val_LSB[p], ksz_sq) + dense_LSB;
                        trg_MSB = Clamp_s32_s32(trg_MSB, 6, 1);     // MSB: 6-bit有符号
                        trg_LSB = Clamp_s32_s32(trg_LSB, 2, 0);     // LSB: 2-bit无符号
                        const int idx = ((i * upscale + p / upscale) * w * upscale + (j * upscale + p % upscale)) * C + k;
                        O[idx] = ((int8_t)trg_MSB << 2) | ((int8_t)trg_LSB); // 直接拼接
                    }
                }
            }
        }
    } else {
        // 获取 pointwise LUT 的偏移量
        int min_LSB[us_sq]; 
        int min_MSB[us_sq];
        int8_t *lut_LSB[us_sq];
        int8_t *lut_MSB[us_sq];
        int max_LSB[us_sq], max_MSB[us_sq];
        int sft_LSB = 0, sft_MSB = 0;
        for(int l = 0; l < us_sq; l++) {
            lut_LSB[l] = &LUT_LSB[sft_LSB * us_sq];
            lut_MSB[l] = &LUT_MSB[sft_MSB * us_sq];
            min_LSB[l] = Round_Div_s32(  0 * (RAT_LSB[l] - offset_LSB), scale_LSB);
            max_LSB[l] = Round_Div_s32(  3 * (RAT_LSB[l] - offset_LSB), scale_LSB);
            min_MSB[l] = Round_Div_s32(-32 * (RAT_MSB[l] - offset_MSB), scale_MSB);
            max_MSB[l] = Round_Div_s32( 31 * (RAT_MSB[l] - offset_MSB), scale_MSB);
            sft_LSB += max_LSB[l] - min_LSB[l] + 1;
            sft_MSB += max_MSB[l] - min_MSB[l] + 1;
            // printf("S: %d, min: %d, max: %d\n", sft_MSB, min_MSB[l], max_MSB[l]);
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < C; k++) {
                    for (int p = 0; p < us_sq; p++) {
                        const int idx = (i * upscale + p / upscale) * (upscale * w) * C + (j * upscale + p % upscale) * C + k;
                        const int lsb = Round_Div_s32((I[idx] & 0x03) * (RAT_LSB[p] - offset_LSB), scale_LSB);
                        const int msb = Round_Div_s32((I[idx] >> 2) * (RAT_MSB[p] - offset_MSB), scale_MSB);
                        I[idx] = (((int8_t)msb << 2) | (int8_t)lsb);
                    }
                }
            }
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < C; k++) {
                    // 同时重置MSB和LSB的中间结果
                    for(int l = 0; l < us_sq; l++) {
                        val_MSB[l] = 0;
                        val_LSB[l] = 0;
                    }
                    // 点卷积累加
                    for (int p = 0; p < us_sq; p++) {
                        const int idx = (i * upscale + p / upscale) * (upscale * w) * C + (j * upscale + p % upscale) * C + k;
                        // const int lsb = Round_Div_s32((I[idx] % 4) * (RAT_LSB[p] - offset_LSB), scale_LSB);
                        // const int msb = Round_Div_s32((I[idx] / 4) * (RAT_MSB[p] - offset_MSB), scale_MSB);
                        // I[idx] = (((int8_t)msb << 2) | (int8_t)lsb);
                        const int idx_LSB = (I[idx] & 0x03) - min_LSB[p];
                        const int idx_MSB = (I[idx] >> 2) - min_MSB[p];
                        for (int l = 0; l < us_sq; l++) {
                            val_LSB[l] += lut_LSB[p][idx_LSB * us_sq + l];
                            val_MSB[l] += lut_MSB[p][idx_MSB * us_sq + l];
                        }
                    }
                    // 结果合并
                    for (int p = 0; p < us_sq; p++) {
                        const int idx = (i * upscale + p / upscale) * (upscale * w) * C + (j * upscale + p % upscale) * C + k;
                        int32_t trg_LSB = Round_Div_s32(val_LSB[p], us_sq);
                        int32_t trg_MSB = Round_Div_s32(val_MSB[p], us_sq);
                        if(dense) {
                            trg_LSB += (I[idx] & 0x03);
                            trg_MSB += (I[idx] >> 2);
                        }
                        if(clamp8) {
                            trg_LSB = Clamp_s32_s32(trg_LSB, 8, 1);
                            trg_MSB = Clamp_s32_s32(trg_MSB, 8, 1);
                            O[idx]  = Clamp_s32_s32(trg_MSB + trg_LSB, 8, 1);
                        } else {
                            trg_LSB = Clamp_s32_s32(trg_LSB, 2, 0);
                            trg_MSB = Clamp_s32_s32(trg_MSB, 6, 1);
                            O[idx]  = (((int8_t)(trg_MSB) << 2) + (int8_t)(trg_LSB));
                        }
                    }
                }
            }
        }
    }
}


void Intp_map_x4(int32_t* O, const int32_t I[16], float upscale) {
    // Map 4x4 Pixel into (1xupscale, 1xupscale) Pixel

}


void Intp_fuse_var_s8_hwc(
    int8_t* O,      // 输出 [H2, W2, C]
    int8_t* I,      // 输入 [H, W, C]
    int8_t* LUT_LSB, 
    int8_t* LUT_MSB,
    int16_t* RAT_LSB,
    int16_t* RAT_MSB,
    int C,
    int H,
    int W,
    int scale_LSB,
    int offset_LSB,
    int scale_MSB,
    int offset_MSB,
    float upscale,
    int ksz,
    int dense,
    int clamp8,
    int dp
) {
    const int pad = 2;
    const int H2 = dp ? (H - pad) : H;
    const int W2 = dp ? (W - pad) : W;
    const float inv_scale = 1.0f / upscale;

    // 清零输出缓冲区
    memset(O, 0, H2 * W2 * C * sizeof(int8_t));

    if(dp) {
        // Depthwise卷积模式 -----------------------------------------------
        const int block_size = (int)ceilf(upscale) + 1;
        #pragma omp parallel for collapse(3)
        for (int i_in = 0; i_in < H - pad; i_in++) {
            for (int j_in = 0; j_in < W - pad; j_in++) {
                for (int k = 0; k < C; k++) {
                    // 生成插值块
                    int32_t val_MSB[block_size*block_size];
                    int32_t val_LSB[block_size*block_size];
                    memset(val_MSB, 0, sizeof(val_MSB));
                    memset(val_LSB, 0, sizeof(val_LSB));

                    // 卷积核遍历
                    for (int m = 0; m < ksz; m++) {
                        for (int n = 0; n < ksz; n++) {
                            const int8_t val = I[((i_in + m) * W + (j_in + n)) * C + k];
                            
                            // LUT索引计算
                            const int idx_MSB_base = (val >> 2) + 32 + (m*ksz + n)*64;
                            const int idx_LSB_base = (val & 0x03) + (m*ksz + n)*4;
                            
                            // 动态插值核
                            for(int dy = 0; dy < block_size; dy++) {
                                for(int dx = 0; dx < block_size; dx++) {
                                    const float ry = dy * inv_scale;
                                    const float rx = dx * inv_scale;
                                    const float wy = 1.0f - fabsf(ry - m);
                                    const float wx = 1.0f - fabsf(rx - n);
                                    const float w = fmaxf(wy * wx, 0.0f);
                                    
                                    val_MSB[dy*block_size + dx] += LUT_MSB[idx_MSB_base] * w;
                                    val_LSB[dy*block_size + dx] += LUT_LSB[idx_LSB_base] * w;
                                }
                            }
                        }
                    }

                    // 将插值块写入输出
                    const int y_out_start = (int)(i_in * upscale);
                    const int x_out_start = (int)(j_in * upscale);
                    for(int dy = 0; dy < block_size; dy++) {
                        for(int dx = 0; dx < block_size; dx++) {
                            const int y_out = y_out_start + dy;
                            const int x_out = x_out_start + dx;
                            if(y_out >= H2 || x_out >= W2) continue;
                            
                            // 归一化并合并结果
                            const int idx = (y_out * W2 + x_out) * C + k;
                            int32_t msb = Round_Div_s32(val_MSB[dy*block_size + dx], ksz*ksz);
                            int32_t lsb = Round_Div_s32(val_LSB[dy*block_size + dx], ksz*ksz);
                            
                            #pragma omp atomic update
                            O[idx] += (msb << 2) | lsb;
                        }
                    }
                }
            }
        }
    } else {
        // Pointwise混合模式 -----------------------------------------------
        #pragma omp parallel for collapse(3)
        for (int i_in = 0; i_in < H; i_in++) {
            for (int j_in = 0; j_in < W; j_in++) {
                for (int k = 0; k < C; k++) {
                    const int8_t val = I[(i_in * W + j_in) * C + k];
                    const int8_t msb = (val >> 2) & 0x3F;
                    const int8_t lsb = val & 0x03;

                    // 计算输出影响区域
                    const int y_out_center = (int)(i_in * upscale);
                    const int x_out_center = (int)(j_in * upscale);
                    const int spread = (int)ceilf(upscale);
                    
                    // 相位相关插值
                    for(int dy = -spread; dy <= spread; dy++) {
                        for(int dx = -spread; dx <= spread; dx++) {
                            const int y_out = y_out_center + dy;
                            const int x_out = x_out_center + dx;
                            if(y_out < 0 || y_out >= H2 || x_out < 0 || x_out >= W2) continue;
                            
                            
                            // 计算相位权重
                            const float ry = fabsf(y_out * inv_scale - i_in);
                            const float rx = fabsf(x_out * inv_scale - j_in);
                            const float w = (1.0f - ry) * (1.0f - rx);
                            if(w < 1e-6f) continue;

                            // 动态选择LUT
                            const int phase_y = (int)(ry * upscale * 2) % 2;
                            const int phase_x = (int)(rx * upscale * 2) % 2;
                            const int phase = phase_y * 2 + phase_x;
                            
                            // 量化参数
                            int rat_lsb = RAT_LSB[phase] - offset_LSB;
                            int rat_msb = RAT_MSB[phase] - offset_MSB;
                            
                            // 结果累加
                            const int idx = (y_out * W2 + x_out) * C + k;
                            int32_t trg = ((msb * rat_msb / scale_MSB) << 2) | 
                                         (lsb * rat_lsb / scale_LSB);
                            
                            #pragma omp atomic update
                            O[idx] += (int8_t)(trg * w);
                        }
                    }
                }
            }
        }
    }

    // 后处理：裁剪和饱和
    #pragma omp parallel for
    for(int i = 0; i < H2*W2*C; i++) {
        if(clamp8) {
            O[i] = Clamp_s32_s32(O[i], 8, 1);
        } else {
            O[i] = Clamp_s32_s32(O[i] >> 2, 6, 1) << 2 | 
                   Clamp_s32_s32(O[i] & 0x03, 2, 0);
        }
    }
}



// ======================================================================== //
//                               Allocator
// ======================================================================== //

void MemInfo_solve(MemInfo info[], size_t nalc, int type) {
    if(!info) {
        return;
    }
    switch (type) {
        case  1: {  // Alloc by lifetime, Set offset order by size

            break;
        }
        case  0:    // Alloc All
        default: {
            int total = 0;
            for(int i = 0; i < nalc; i++) {
                info[i].offset = total;
                total += info[i].size;
            }
            break;
        }
    }
}


void MemPool_init(MemPool* pool, int nalc, MemInfo info[]) {
    if(!pool) {
        return;
    }
    pool->type = -1;
    pool->nalc = nalc;
    pool->info = info;
    pool->size = 0;
    pool->mem = NULL;
}

void MemPool_alc(MemPool* pool) {
    if(!pool) {
        return;
    }
    MemInfo* info = pool->info;
    // find max (offset + size)
    size_t max = 0;
    for(int i = 0; i < pool->nalc; i++) {
        if(max < info[i].offset + info[i].size) {
            max = info[i].offset + info[i].size;
        }
    }
    if(max == 0 || pool->size == max) {
        return;
    }
    if(pool->mem) {
        free(pool->mem);
    }
    OPR_ERR("MemPool alloc size: %ld B\n", max);
    pool->size = max;
    pool->mem = (uint8_t*)malloc(pool->size);
}

void* MemPool_get(MemPool* pool, int idx) {
    if(idx >= pool->nalc) {
        return NULL;
    }
    return pool->mem + pool->info[idx].offset;
}

void MemPool_solve(MemPool* pool, void** ptr, int type) {
    if(!pool) {
        return;
    }
    MemInfo_solve(pool->info, pool->nalc, type);
    pool->type = type;
    // alloc & map to ptr
    if(ptr) {
        MemPool_alc(pool);
        for(int i = 0; i < pool->nalc; i++) {
            ptr[i] = MemPool_get(pool, i);
        }
    }
}

void MemPool_disp(MemPool* pool) {
    if(!pool) {
        return;
    }
    OPR_ERR("|%d|Pool        Address       Start     End             Size         Offset\n", pool->type);
    for(int i = 0; i < pool->nalc; i++) {
        OPR_ERR("- [%3d]:  %16p    %4d -> %4d    <%12ld + %12ld>\n", i, pool->mem == NULL ? NULL : pool->mem + pool->info[i].offset, pool->info[i].start, pool->info[i].end, pool->info[i].size, pool->info[i].offset);
    }
}


void MemPool_free(MemPool* pool) {
    if(!pool) {
        return;
    }
    if(pool->mem) {
        free(pool->mem);
        OPR_ERR("MemPool free size: %ld B\n", pool->size);
    }
    pool->mem = NULL;
    pool->size = 0;
}


