/**
 * Module       : DepthLUT
 * Input        : I [C][H][W] 
 * Output       : O [C][H_OUT][W_OUT]
 * Description  : Depthwise Interpolation Lookup Table
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    C           = context.get('C', 3)              # 通道数
    H           = context.get('H', 48)             # 输入高度
    W           = context.get('W', 48)             # 输入宽度
    UPSCALE     = context.get('UPSCALE', 2)        # 上采样倍数
    KSZ         = context.get('KSZ', 3)            # 卷积核尺寸
    DW          = context.get('DW', 8)             # 数据位宽
    
    LSB_BITS    = 2
    MSB_BITS    = DW - LSB_BITS
    US_SQ       = UPSCALE * UPSCALE
    KSZ_SQ      = KSZ * KSZ
    H_OUT       = (H - (KSZ-1)) / UPSCALE         # 输出高度
    W_OUT       = (W - (KSZ-1)) / UPSCALE         # 输出宽度
    LSB_CLAMP   = f"U{LSB_BITS}"
    MSB_CLAMP   = f"S{MSB_BITS}"
    KER_RODIV   = f"P{KSZ_SQ}"

%>
module dw_intp_fuse #(
    parameter C = ${C},       // 通道数
    parameter H = ${H},       // 输入高度
    parameter W = ${W},       // 输入宽度
    parameter DW = ${DW}      // 数据位宽
)(
    // input  logic clk,
    // input  logic rst_n,
    // 数据接口
    input  logic signed [DW-1:0] I [0:C-1][0:H-1][0:W-1],                       // 输入张量
    output logic signed [DW-1:0] O [0:C-1][0:H_OUT-1][0:W_OUT-1][0:US_SQ-1],    // 输出张量
    // LUT接口
    input  logic [DW-1:0] LUT_LSB [0:${(1<<LSB_BITS)*KSZ_SQ}-1][0:US_SQ-1],
    input  logic [DW-1:0] LUT_MSB [0:${(1<<MSB_BITS)*KSZ_SQ}-1][0:US_SQ-1]
);

// 组合逻辑计算核心
generate
    for (genvar c = 0; c < C; c++) begin : CHAN
        for (genvar h = 0; h < H_OUT; h++) begin : H_ROW
            for (genvar w = 0; w < W_OUT; w++) begin : W_COL
                // 当前处理窗口坐标
                localparam int HI = h * UPSCALE;
                localparam int WI = w * UPSCALE;
                
                // 组合逻辑卷积窗口
                logic signed [DW-1:0] kernel [0:KSZ-1][0:KSZ-1];
                always_comb begin
                    for (int i = 0; i < KSZ; i++) begin
                        for (int j = 0; j < KSZ; j++) begin
                            kernel[i][j] = I[c][HI+i][WI+j];
                        end
                    end
                end

                // 并行插值点生成
                for (genvar p = 0; p < US_SQ; p++) begin : UPSAMPLE
                    // 累加器组合逻辑
                    logic signed [31:0] acc_msb, acc_lsb;
                    always_comb begin
                        acc_msb = 0;
                        acc_lsb = 0;
                        % for m in range(KSZ):
                        % for n in range(KSZ):
                        begin
                            automatic logic [MSB_BITS-1:0] msb_part = 
                                kernel[${m}][${n}][DW-1:DW-MSB_BITS];
                            automatic logic [LSB_BITS-1:0] lsb_part = 
                                kernel[${m}][${n}][LSB_BITS-1:0];
                            
                            acc_msb += $signed(LUT_MSB[${m*KSZ + n}*(1<<MSB_BITS) + msb_part][p]);
                            acc_lsb += $signed(LUT_LSB[${m*KSZ + n}*(1<<LSB_BITS) + lsb_part][p]);
                        end
                        % endfor
                        % endfor
                    end

                    // 组合量化逻辑
                    logic signed [31:0] div_msb, div_lsb;
                    RoundDivS32_${KER_RODIV} u_div_msb (
                        .x(acc_msb),
                        .result(div_msb)
                    );
                    
                    RoundDivS32_${KER_RODIV} u_div_lsb (
                        .x(acc_lsb),
                        .result(div_lsb)
                    );

                    // 组合钳位逻辑
                    logic signed [MSB_BITS-1:0] clamp_msb;
                    logic signed [LSB_BITS-1:0] clamp_lsb;
                    ClampS32_${MSB_CLAMP} u_clamp_msb (
                        .x(div_msb),
                        .y(clamp_msb)
                    );
                    
                    ClampS32_${LSB_CLAMP} u_clamp_lsb (
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
