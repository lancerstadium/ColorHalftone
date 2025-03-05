/**
 * Module       : RoundDivS32
 * Input        : x[31:0] - Signed 32-bit dividend
 *              : y[31:0] - Signed 32-bit divisor
 * Output       : result[31:0] - Signed 32-bit quotient
 * Description  : Signed 32-bit division with rounding to nearest integer.
 * Author       : lancerstadium
 * Date         : Tue Mar  4 16:41:09 CST 2025
 * License      : MIT
 */

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
module RoundDivS32_P9 (
    input  logic signed [31:0] x,
    output logic signed [31:0] result
);
    // 通用倒数乘法实现
    localparam RECIPROCAL = 1073741824;
    always_comb begin
        logic signed [63:0] scaled;
        logic signed [31:0] base_adj = 9 >> 1;
        logic signed [31:0] adj = base_adj;
        
        if (9 == 0) begin
            result = 0;
        end else begin
            scaled = (x + adj) * RECIPROCAL;
            // 符号校正与精度对齐
            result = (x[31] ^ 0) ? 
                    -(scaled[61:30]) : scaled[61:30];
        end
    end
endmodule
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on WIDTHEXPAND */
