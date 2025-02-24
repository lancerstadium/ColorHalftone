#ifndef SR_LUT_LUT_H
#define SR_LUT_LUT_H

#include "data.h"
#include "intp.h"

#define LUT_PTR_MAX 6
#define LUT_SCALE_QUANT 16
#define LUT_PERF 1
#define LUT_MEMPOOL 1

#define CONCAT(a, b) a ## b
#define CONCAT3(a, b, c) a ## b ## c
#define CONCAT4(a, b, c, d) a ## b ## c ## d
#define CONCAT5(a, b, c, d, e) a ## b ## c ## d ## e
#define LUT_OPR(op, dtype) CONCAT3(op,_,dtype)

#if LUT_PERF
#include <sys/time.h>
static inline double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

// ======================================================================== //
//                               MuLUT
// ======================================================================== //

typedef struct {
    char* dir;
    char* modes;
    int nmode;
    int nstage;
    int bitwidth;
    int interval;
    int scale;
    int free_mid;
    TensorData *lut[LUT_PTR_MAX];
    int32_t *mid[8];
} MuLUT;

void MuLUT_init(MuLUT *mdl, char* dir, char* modes, int nmode, int nstage, int bitwidth, int interval, int scale);
void MuLUT_free(MuLUT *mdl);
void MuLUT_forward(MuLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C);


// ======================================================================== //
//                               TinyLUT
// ======================================================================== //

static float TinyLUT_hl1_f[] = { 0.7511365413665771, 0.8216833472251892, 0.0000994513975456357, 0.5006141066551208, -0.00004777031790581532, 0.8335879445075989, 0.000041454561142018065, 0.0008207048522308469, 0.48277968168258667, 0.00016744111781008542, 0.000018090317098540254, -0.0001536521449452266, 0.7519780397415161, -0.0000873714016051963, 1.1355959177017212, 0.00010884129005717114 };
static float TinyLUT_hh1_f[] = { 0.8757619857788086, 0.9348493218421936, 0.952409565448761, 0.822011411190033, 0.8541820645332336, 0.8391884565353394, 0.7854822874069214, 0.9182155132293701, 0.882203996181488, 0.9594336152076721, 0.9280604124069214, 1.2542303800582886, 0.8856714963912964, 0.9604812860488892, 0.875754177570343, 0.928962230682373 };
static float TinyLUT_hl2_f[] = { 0.5862683057785034, 0.8336194157600403, 0.1665961891412735, 0.5662277340888977, 0.527957022190094, 0.12872901558876038, 0.44153282046318054, 0.30386894941329956, 0.49993911385536194, 0.4918636977672577, 0.16676542162895203, 0.16643026471138, 0.7525836229324341, 0.16646282374858856, 0.37890106439590454, 0.3320236802101135 };
static float TinyLUT_hh2_f[] = { 0.7246444225311279, 0.8162808418273926, 0.8764151334762573, 0.9065658450126648, 0.6104913353919983, 0.5581252574920654, 0.7327605485916138, 0.5743480324745178, 0.7503433227539062, 0.5982151031494141, 0.9173057675361633, 0.6895579695701599, 0.9217936396598816, 0.5010436177253723, 0.8508085012435913, 0.6147695183753967 };

#if LUT_SCALE_QUANT == 8

static int TinyLUT_hl1_scale  = 111;
static int TinyLUT_hl1_offset = 0;
static int8_t TinyLUT_hl1[] = { 84,92,0,56,0,93,0,0,54,0,0,0,84,0,127,0 };

static int TinyLUT_hh1_scale  = 270;
static int TinyLUT_hh1_offset = -212;
static int8_t TinyLUT_hh1[] = { 25,41,46,11,19,15,1,37,27,48,39,127,28,48,25,40 };

static int TinyLUT_hh2_scale  = 301;
static int TinyLUT_hh2_offset = -151;
static int8_t TinyLUT_hh2[] = { 68,95,114,123,33,17,70,22,75,30,126,57,127,0,106,35 };

#elif LUT_SCALE_QUANT == 16

static int TinyLUT_hl1_scale  = 28850;
static int TinyLUT_hl1_offset = 4;
static int16_t TinyLUT_hl1[] = { 21675,23710,7,14447,2,24053,5,28,13932,9,5,0,21699,0,32767,7 };

static int TinyLUT_hh1_scale  = 69903;
static int TinyLUT_hh1_offset = -54907;
static int16_t TinyLUT_hh1[] = { 6312,10442,11669,2554,4803,3755,1,9279,6762,12161,9967,32767,7004,12234,6311,10030 };

static int TinyLUT_hl2_scale  = 46485;
static int TinyLUT_hl2_offset = -5983;
static int16_t TinyLUT_hl2[] = { 21270,32767,1761,20338,18559,1,14542,8142,17257,16881,1769,1754,29001,1755,11630,9451 };

static int TinyLUT_hh2_scale  = 77877;
static int TinyLUT_hh2_offset = -39020;
static int16_t TinyLUT_hh2[] = { 17414,24550,29233,31581,8524,4445,18046,5709,19415,7568,32418,14681,32767,0,27239,8857 };

#endif // LUT_SCALE_QUANT

typedef struct {
    char* dir;
    char* modes;
    int nmode;
    int nstage;
    int bitwidth;
    int interval;
    int scale;
    int free_mid;
    TensorData *lut[LUT_PTR_MAX];
    void *mid[10];
#ifdef LUT_MEMPOOL
    MemPool mem_pool;
#endif // LUT_MEMPOOL
#ifdef LUT_PERF
    int perf_cnt;
    float perf_time;
    float perf_fps;
    size_t perf_mem;
#endif // LUT_PERF
} TinyLUT;

void TinyLUT_init(TinyLUT *mdl, char* dir, char* modes, int nmode, int nstage, int bitwidth, int interval, int scale);
void TinyLUT_free(TinyLUT *mdl);
void TinyLUT_forward(TinyLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C);
void TinyLUT_forward_opt(TinyLUT *mdl, uint8_t* O, uint8_t* I, int H, int W, int C);

#endif // SR_LUT_LUT_H