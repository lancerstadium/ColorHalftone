#include "lut.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



void MuLUT_test() {
    int total = 10;
    char* IMG[] = {
        "0149x4",
        "0152x4",
        "0261x4",
        "0399x4",
        "0406x4",
        "0509x4",
        "0582x4",
        "0693x4",
        "0723x4",
        "0740x4"
    };

    char img_path[64];
    int W, H, C, scale = 4;
    uint8_t *X = (uint8_t *)stbi_load("../../test/org/0149x4.png", &W, &H, &C, 0);
    int H2 = H * scale;
    int W2 = W * scale;
    uint8_t *Y = (uint8_t *)calloc(H2 * W2 * C, sizeof(uint8_t));

    MuLUT mdl;
    MuLUT_init(&mdl, "../../lut/MuLUT", "sdy", 3, 2, 8, 4, scale);
    mdl.free_mid = 0;

    for (int i = 0; i < total; i++) {
        sprintf(img_path, "../../test/org/%s.png", IMG[i]);
        X = (uint8_t *)stbi_load(img_path, &W, &H, &C, 0);
        MuLUT_forward(&mdl, Y, X, H, W, C);
        // Cast_s32_u8(X, mdl.mid[0], H * W * C);
        sprintf(img_path, "../../test/c/%s.bmp", IMG[i]);
        stbi_write_bmp(img_path, W2, H2, C, Y);
        // stbi_write_bmp(img_path, W, H, C, X);
        printf("img %s done\n", IMG[i]);
    }

    MuLUT_free(&mdl);
}

#define USE_OPTIMIZED

void TinyLUT_test() {
    int total = 10;
    char* IMG[] = {
        "0149x4",
        "0152x4",
        "0261x4",
        "0399x4",
        "0406x4",
        "0509x4",
        "0582x4",
        "0693x4",
        "0723x4",
        "0740x4"
    };

    char img_path[64];
    int W, H, C, scale = 4;
    uint8_t *X = (uint8_t *)stbi_load("../../test/org/0149x4.png", &W, &H, &C, 0);
    int H2 = H * scale;
    int W2 = W * scale;
    uint8_t *Y = (uint8_t *)calloc(H2 * W2 * C, sizeof(uint8_t));

    TinyLUT mdl;
    TinyLUT_init(&mdl, "../../lut/TinyLUT", "DPU", 3, 3, 8, 4, 4);

    for (int i = 0; i < total; i++) {
        sprintf(img_path, "../../test/org/%s.png", IMG[i]);
        X = (uint8_t *)stbi_load(img_path, &W, &H, &C, 0);
#if   defined(USE_BICUBIC)
        Intp_bicubic_s32_hwc(Y, X, C, H, W, 4);
#elif defined(USE_OPTIMIZED)
        TinyLUT_forward_opt(&mdl, Y, X, H, W, C);
#else
        TinyLUT_forward(&mdl, Y, X, H, W, C);
#endif
        sprintf(img_path, "../../test/c1/%s.bmp", IMG[i]);
        stbi_write_bmp(img_path, W2, H2, C, Y);
        printf("img %s done\n", IMG[i]);
    }

    TinyLUT_free(&mdl);
}

void Quant_s8_test() {
    int scale = 0, offset = 0;
    int8_t Qout[16];
    float* trg = TinyLUT_hl1_f;
    Disp_f32_chw(trg, 16, 1, 1);
    Quant_s8_f32(Qout, trg, 16, &scale, &offset);
    Disp_s8_chw(Qout, 16, 1, 1);
    Dequant_s8_f32(trg, Qout, 16, scale, offset);
    Disp_f32_chw(trg, 16, 1, 1);
    printf("scale = %d, offset = %d\n", scale, offset);
}

void Quant_s16_test() {         // Error
    int scale = 0, offset = 0;
    int16_t Qout[16];
    float* trg = TinyLUT_hl1_f;
    Disp_f32_chw(trg, 16, 1, 1);
    Quant_s16_f32(Qout, trg, 16, &scale, &offset);
    // Disp_s16_chw(Qout, 16, 1, 1);
    for(int i = 0; i < 16; i++) {
        printf("%d,", Qout[i]);
    }
    printf("\n");
    Dequant_s16_f32(trg, Qout, 16, scale, offset);
    Disp_f32_chw(trg, 16, 1, 1);
    printf("scale = %d, offset = %d\n", scale, offset);
}

int main() {
    // MuLUT_test();
    // Quant_s16_test();
    TinyLUT_test();
    return 0;
}
