/**
 * Module       : RoundDivS32
 * Input        : x[31:0] - Signed 32-bit dividend
 *              : y[31:0] - Signed 32-bit divisor
 * Output       : result[31:0] - Signed 32-bit quotient
 * Description  : Signed 32-bit division with rounding to nearest integer.
 * Author       : lancerstadium
 * Date         : Sun Mar  2 18:12:13 CST 2025
 * License      : MIT
 */

/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
module RoundDivS32_P16 (
    input  logic signed [31:0] x,
    output logic signed [31:0] result
);
    // 静态2^n优化路径
    localparam SHIFT = 4;
    always_comb begin
        logic signed [31:0] adj = 16 >> 1;
        if (16 == 0) begin
            result = 0;
        end else begin
            // 符号统一处理
            logic signed [32:0] adjusted = x + (adj);
            result = adjusted >>> SHIFT;
        end
    end
endmodule
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on WIDTHEXPAND */
