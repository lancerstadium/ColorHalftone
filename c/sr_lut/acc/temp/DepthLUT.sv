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
    import math

    # 计算地址位宽
    def clog2(x):
        return math.ceil(math.log2(x)) if x >0 else 0

    C           = context.get('C', 3)              # 通道数
    H           = context.get('H', 50)             # 输入高度
    W           = context.get('W', 50)             # 输入宽度
    UPSCALE     = context.get('UPSCALE', 2)        # 上采样倍数
    KSZ         = context.get('KSZ', 3)            # 卷积核尺寸
    DW          = context.get('DW', 8)             # 数据位
    msb_name    = context.get('msb_name', '')
    lsb_name    = context.get('lsb_name', '')
    
    LSB_BITS    = 2
    MSB_BITS    = DW - LSB_BITS
    US_SQ       = UPSCALE * UPSCALE
    KSZ_SQ      = KSZ * KSZ
    PAD         = KSZ - 1
    H_OUT       = int((H - PAD))                   # 输出高度
    W_OUT       = int((W - PAD))                   # 输出宽度
    LSB_CLAMP   = f"U{LSB_BITS}"
    MSB_CLAMP   = f"S{MSB_BITS}"
    KER_RODIV   = f"P{KSZ_SQ}"

    LUT_BATCH   = KSZ_SQ
    LUT_ADDRW_MSB = clog2((1<<MSB_BITS)*KSZ_SQ)
    LUT_ADDRW_LSB = clog2((1<<LSB_BITS)*KSZ_SQ)

    module_name = f"DepthLUT_{C}x{H}x{W}_K{KSZ}_U{UPSCALE}_D{DW}"
%>
module ${module_name} (
    input  logic         clk,
    input  logic         rst_n,
    input  logic signed [${DW-1}:0] I [0:${C-1}][0:${H-1}][0:${W-1}],
    output logic signed [${DW-1}:0] O [0:${C-1}][0:${H_OUT-1}][0:${W_OUT-1}][0:${US_SQ-1}],
    // LUT模块接口
    output logic        lut_req,      // 同时触发两个LUT
    input  logic        lut_done     // 两个LUT都完成
);

    //============= LUT模块信号定义 =============//
    logic [${LUT_ADDRW_MSB-1}:0] lut_addr_msb_base;
    logic [${LUT_ADDRW_LSB-1}:0] lut_addr_lsb_base;
    logic signed [7:0] lut_data_msb [0:${LUT_BATCH-1}];
    logic signed [7:0] lut_data_lsb [0:${LUT_BATCH-1}];
    logic msb_done, lsb_done;

    //============= LUT模块实例化 =============//
    LUTTable_${msb_name} u_lut_msb (
        .clk(clk),
        .rst_n(rst_n),
        .burst_start(lut_req),
        .burst_done(msb_done),
        .base_addr(lut_addr_msb_base),
        .data(lut_data_msb)
    );

    LUTTable_${lsb_name} u_lut_lsb (
        .clk(clk),
        .rst_n(rst_n),
        .burst_start(lut_req),
        .burst_done(lsb_done),
        .base_addr(lut_addr_lsb_base),
        .data(lut_data_lsb)
    );

    assign lut_done = msb_done & lsb_done;

    //============= 核心计算逻辑优化 =============//
    generate
        for (genvar c = 0; c < ${C}; c++) begin : CHAN
            for (genvar h = 0; h < ${H_OUT}; h++) begin : H_ROW
                for (genvar w = 0; w < ${W_OUT}; w++) begin : W_COL
                    // 窗口缓存寄存器
                    logic signed [${DW-1}:0] kernel_reg [0:${KSZ_SQ-1}];
                    logic kernel_valid;
                    logic [${clog2(US_SQ)-1}:0] phase_cnt;
                    
                    // 修正后的地址生成逻辑
                    always_ff @(posedge clk) begin
                        if (!rst_n) begin
                            kernel_valid <= 0;
                            phase_cnt <= 0;
                            lut_req <= 0;
                            // 初始化地址寄存器
                            lut_addr_msb_base <= 0;
                            lut_addr_lsb_base <= 0;
                        end else begin
                            // 装载新kernel
                            % for i in range(KSZ_SQ):
                            kernel_reg[${i}] <= I[c][h*${UPSCALE}+${i//KSZ}][w*${UPSCALE}+${i%KSZ}];
                            % endfor
                            kernel_valid <= 1;

                            // 地址生成逻辑（修正拼接顺序）
                            if (kernel_valid) begin
                                % for i in range(KSZ_SQ):
                                lut_addr_msb_base[${i}*${MSB_BITS} +: ${MSB_BITS}] <= 
                                    kernel_reg[${i}][${DW-1}:${DW-MSB_BITS}];
                                lut_addr_lsb_base[${i}*${LSB_BITS} +: ${LSB_BITS}] <= 
                                    kernel_reg[${i}][${LSB_BITS-1}:0];
                                % endfor
                                lut_req <= 1;
                            end else begin
                                lut_req <= 0;
                            end
                        end
                    end

                    // 恢复原始后处理模块
                    for (genvar p = 0; p < ${US_SQ}; p++) begin : PHASE
                        logic signed [31:0] acc_msb, acc_lsb;
                        logic signed [31:0] div_msb, div_lsb;
                        logic signed [${MSB_BITS-1}:0] clamp_msb;
                        logic signed [${LSB_BITS-1}:0] clamp_lsb;

                        // 累加逻辑
                        always_ff @(posedge clk) begin
                            if (lut_done) begin
                                acc_msb <= 0;
                                acc_lsb <= 0;
                                % for k in range(KSZ_SQ):
                                acc_msb <= acc_msb + $signed(lut_data_msb[${k}]);
                                acc_lsb <= acc_lsb + $signed(lut_data_lsb[${k}]);
                                % endfor
                            end
                        end

                        // 恢复原始除法模块
                        RoundDivS32_${KER_RODIV} u_div_msb (
                            .clk(clk),
                            .x(acc_msb),
                            .result(div_msb)
                        );
                        
                        RoundDivS32_${KER_RODIV} u_div_lsb (
                            .clk(clk),
                            .x(acc_lsb),
                            .result(div_lsb)
                        );

                        // 恢复原始钳位模块
                        ClampS32_${MSB_CLAMP} u_clamp_msb (
                            .clk(clk),
                            .x(div_msb),
                            .y(clamp_msb)
                        );
                        
                        ClampS32_${LSB_CLAMP} u_clamp_lsb (
                            .clk(clk),
                            .x(div_lsb),
                            .y(clamp_lsb)
                        );

                        // 输出寄存器
                        always_ff @(posedge clk) begin
                            O[c][h][w][p] <= {clamp_msb, clamp_lsb};
                        end
                    end
                end
            end
        end
    endgenerate

endmodule
