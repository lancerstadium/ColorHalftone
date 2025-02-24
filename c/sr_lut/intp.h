/**
 * 
 * ## Functions
 * 
 * - [*] Lut_pad_dict
 * - [*] Disp_u8_hwc
 * - [*] Disp_u8_chw
 * 
 * 
 * ## Operators
 * 
 * - Data format: HWC
 * 
 * - [*] Memory     <s32,u8>    : S/DAlc, Copy, DRlz
 * - [*] Set        <s32,u8>    : Zero, Fill
 * - [*] Mul        <s32,u8>    : Mul
 * - [*] Add        <s32,u8>    : Add, Addi
 * - [*] Cast       <s32,u8>    : Cast
 * - [*] Clamp      <s32,u8>    : Clamp
 * - [*] Quant          <s8>    : Quant, Dequant
 * - [*] Pad        <s32,u8>    : Pad
 * - [*] Rotate     <s32,u8>    : Rot
 * - [*] Segmentation  <s32>    : Seg
 * - [*] Interpolation <s32>    : Intp_simplex4d, Intp_depthwise, Intp_pointwise
 * 
 * 
 * ## Allocators
 * 
 */

#ifndef SR_LUT_INTP_H
#define SR_LUT_INTP_H

#include <stdint.h>
#include <stddef.h>

// ======================================================================== //
//                               Functions
// ======================================================================== //


int Lut_pad_dict(char mode);

void Disp_u8_hwc(uint8_t *x, int H, int W, int C);
void Disp_u32_hwc(uint32_t *x, int H, int W, int C);
void Disp_s32_hwc(int32_t *x, int H, int W, int C);
void Disp_u8_chw(uint8_t *x, int H, int W, int C);
void Disp_u32_chw(uint32_t *x, int H, int W, int C);
void Disp_s32_chw(int32_t *x, int H, int W, int C);
void Disp_s16_chw(int16_t *x, int H, int W, int C);
void Disp_s8_chw(int8_t *x, int H, int W, int C);
void Disp_f32_chw(float *x, int H, int W, int C);

// ======================================================================== //
//                               Operators
// ======================================================================== //


#define SAlc_s32(S, N) static int32_t S[N]
#define SAlc_u8(S, N) static uint8_t S[N]

int32_t* DAlc_s32(int N);
uint8_t* DAlc_u8(int N);
int8_t* DAlc_s8(int N);
void Copy_s32(int32_t* O, int32_t* I, int N);
void Copy_u8(uint8_t* O, uint8_t* I, int N);
void DRlz_s32(int32_t* x);
void DRlz_u8(uint8_t* x);
void DRlz_s8(int8_t* I);
void Zero_s32(int32_t* O, int N);
void Zero_u8(uint8_t* O, int N);
void Zero_s8(int8_t* O, int N);
void Fill_s32(int32_t* O, int N, int32_t val);
void Fill_u8(uint8_t* O, int N, uint8_t val);
void Add_s32(int32_t* O, int32_t* I1, int32_t* I2, int N);
void Add_u8(uint8_t* O, uint8_t* I1, uint8_t* I2, int N);
void Add_s8(int8_t* O, int8_t* I1, int8_t* I2, int N);
void Addi_s32(int32_t* O, int32_t* I1, int32_t I2, int N);
void Addi_u8(uint8_t* O, uint8_t* I1, uint8_t I2, int N);
void Mul_s32(int32_t* O, int32_t* I1, int32_t* I2, int N);
void Mul_u8(uint8_t* O, uint8_t* I1, uint8_t* I2, int N);
void MulQ_tile_f32_s32_hwc(int32_t* O, int32_t* I1, float* I2, int C, int H, int W, int sz);
void MulQ_tile_s8_s32_hwc(int32_t* O, int32_t* I1, int8_t* I2, int scale, int offset, int C, int H, int W, int sz);
void MulQ_tile_s16_s32_hwc(int32_t* O, int32_t* I1, int16_t* I2, int scale, int offset, int C, int H, int W, int sz);
void MulQ_tile_s16_s8_hwc(int8_t* O, int8_t* I1, int16_t* I2, int scale, int offset, int C, int H, int W, int sz);
uint8_t Clamp_u8_s32(int32_t x);
void Clamp_s32(int32_t* O, int32_t* I, int N, int bitwidth, int sign);
void Clamp_s8(int8_t* O, int8_t* I, int N, int bitwidth, int sign);
void Quant_s8_f32(int8_t* O, float* I, int N, int* scale, int* offset);
void Quant_s16_f32(int16_t* O, float* I, int N, int* scale, int* offset);
void Dequant_s8_f32(float* O, int8_t* I, int N, int scale, int offset);
void Dequant_s16_f32(float* O, int16_t* I, int N, int scale, int offset);
void Cast_s32_u8(int32_t* O, uint8_t* I, int N);
void Cast_u8_s32(uint8_t* O, int32_t* I, int N);
void Cast_s8_u8(int8_t* O, uint8_t* I, int N);
void Cast_u8_s8(uint8_t* O, int8_t* I, int N);
void Pad_u8_hwc(uint8_t* O, uint8_t* I, int C, int H, int W, int pad);
void Pad_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int pad, int rb, int reflect);
void Pad_s8_hwc(int8_t* O, int8_t* I, int C, int H, int W, int pad, int rb, int reflect);
void Crop_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int pad, int rb);
void Rot_u8_hwc(uint8_t* O, uint8_t* I, int C, int H, int W, int times);
void Rot_s32_hwc(int32_t* O, int32_t* I, int C, int H, int W, int times);
void Rot_s8_hwc(int8_t* O, int8_t* I, int C, int H, int W, int times);
void Seg_s32_hwc(int32_t* O1, int32_t* O2, int32_t* I, int N, int interval);
void Seg_u8_hwc(uint8_t* O1, uint8_t* O2, uint8_t* I, int N, int interval);
void Seg_s8_hwc(int8_t* O1, int8_t* O2, int8_t* I, int N, int interval);


void Intp_bicubic_s32_hwc(
    int32_t* O, 
    int32_t* I, 
    int C, 
    int H, 
    int W, 
    int upscale
);

void Intp_simplex4d_s32_hwc(
    int32_t* O,     
    int32_t* I,     
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
);

void Intp_depthwise_s32_hwc(
    int32_t* O,     
    int32_t* I,     
    int8_t* LUT,    // -127 ~ 127
    int C,
    int H,
    int W,          
    int ksz,        // 3
    int upscale,    // 4
    int dense,      // if dense
    int hl          // high(1) | low (0)
);


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
    int clamp8,     // if clamp8
    int hl          // high(1) | low (0)
);


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
);


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
);


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
);

// ======================================================================== //
//                               Allocators
// ======================================================================== //

typedef struct {
    int start;
    int end;
    size_t size;            // Base: 1B
    size_t offset;          // Base: 1B
} MemInfo;

void MemInfo_solve(MemInfo info[], size_t nalc, int type);

typedef struct {
    void* mem;
    size_t size;            // Base: 1B
    int  nalc;
    int  type;              // Solve type
    MemInfo* info;
} MemPool;


void MemPool_init(MemPool* pool, int nalc, MemInfo info[]);
void* MemPool_get(MemPool* pool, int idx);
void MemPool_alc(MemPool* pool);
void MemPool_disp(MemPool* pool);
void MemPool_solve(MemPool* pool, void** ptr, int type);
void MemPool_free(MemPool* pool);


#endif // SR_LUT_INTP_H