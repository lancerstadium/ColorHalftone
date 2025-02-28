/**
 * Module       : DepthLUT
 * Input        : I [C][H][W] 
 * Output       : O [C][H_OUT][W_OUT]
 * Description  : Depthwise Interpolation Lookup Table
 * Author       : lancerstadium
 * Date         : Sat Mar  1 00:00:00 CST 2025
 * License      : MIT
 */

module DepthLUT_3x50x50_K3_U4_D8 (
    input  logic signed [7:0] I [0:2][0:49][0:49],                   // 输入张量
    output logic signed [7:0] O [0:2][0:11][0:11][0:15],     // 输出张量
    input  logic signed [7:0] LUT_LSB [0:35][0:15],
    input  logic signed [7:0] LUT_MSB [0:575][0:15]
);
generate
    for (genvar c = 0; c < 3; c++) begin : CHAN
        for (genvar h = 0; h < 12; h++) begin : H_ROW
            for (genvar w = 0; w < 12; w++) begin : W_COL
                // 当前处理窗口坐标
                localparam int HI = h * 4;
                localparam int WI = w * 4;
                
                // 组合逻辑卷积窗口
                logic signed [7:0] kernel [0:2][0:2];
                always_comb begin
                    kernel[0][0] = I[c][HI+0][WI+0];
                    kernel[0][1] = I[c][HI+0][WI+1];
                    kernel[0][2] = I[c][HI+0][WI+2];
                    kernel[1][0] = I[c][HI+1][WI+0];
                    kernel[1][1] = I[c][HI+1][WI+1];
                    kernel[1][2] = I[c][HI+1][WI+2];
                    kernel[2][0] = I[c][HI+2][WI+0];
                    kernel[2][1] = I[c][HI+2][WI+1];
                    kernel[2][2] = I[c][HI+2][WI+2];
                end

                // 并行插值点生成
                for (genvar p = 0; p < 16; p++) begin : UPSAMPLE
                    // 累加器组合逻辑
                    logic signed [31:0] acc_msb, acc_lsb;
                    always_comb begin
                        acc_msb = 0;
                        acc_lsb = 0;
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[0][0][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[0][0][1:0];
                            
                            acc_msb += LUT_MSB[0 + msb_part][p];
                            acc_lsb += LUT_LSB[0 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[0][1][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[0][1][1:0];
                            
                            acc_msb += LUT_MSB[64 + msb_part][p];
                            acc_lsb += LUT_LSB[4 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[0][2][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[0][2][1:0];
                            
                            acc_msb += LUT_MSB[128 + msb_part][p];
                            acc_lsb += LUT_LSB[8 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[1][0][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[1][0][1:0];
                            
                            acc_msb += LUT_MSB[3 + msb_part][p];
                            acc_lsb += LUT_LSB[3 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[1][1][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[1][1][1:0];
                            
                            acc_msb += LUT_MSB[67 + msb_part][p];
                            acc_lsb += LUT_LSB[7 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[1][2][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[1][2][1:0];
                            
                            acc_msb += LUT_MSB[131 + msb_part][p];
                            acc_lsb += LUT_LSB[11 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[2][0][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[2][0][1:0];
                            
                            acc_msb += LUT_MSB[6 + msb_part][p];
                            acc_lsb += LUT_LSB[6 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[2][1][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[2][1][1:0];
                            
                            acc_msb += LUT_MSB[70 + msb_part][p];
                            acc_lsb += LUT_LSB[10 + lsb_part][p];
                        end
                        begin
                            automatic logic [5:0] msb_part = 
                                kernel[2][2][7:2];
                            automatic logic [1:0] lsb_part = 
                                kernel[2][2][1:0];
                            
                            acc_msb += LUT_MSB[134 + msb_part][p];
                            acc_lsb += LUT_LSB[14 + lsb_part][p];
                        end
                    end

                    // 组合量化逻辑
                    logic signed [31:0] div_msb, div_lsb;
                    RoundDivS32_P9 u_div_msb (
                        .x(acc_msb),
                        .result(div_msb)
                    );
                    
                    RoundDivS32_P9 u_div_lsb (
                        .x(acc_lsb),
                        .result(div_lsb)
                    );

                    logic signed [5:0] clamp_msb;
                    logic signed [1:0] clamp_lsb;
                    ClampS32_S6 u_clamp_msb (
                        .x(div_msb),
                        .y(clamp_msb)
                    );
                    
                    ClampS32_U2 u_clamp_lsb (
                        .x(div_lsb),
                        .y(clamp_lsb)
                    );

                    // 最终输出组合
                    assign O[c][h][w][p] = {clamp_msb, clamp_lsb};
                end
            end
        end
    end
endgenerate

endmodule
