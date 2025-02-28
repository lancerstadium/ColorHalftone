/**
 * Module       : RoundDivS32
 * Input        : x[31:0] - Signed 32-bit dividend
 *              : y[31:0] - Signed 32-bit divisor
 * Output       : result[31:0] - Signed 32-bit quotient
 * Description  : Signed 32-bit division with rounding to nearest integer.
 * Author       : lancerstadium
 * Date         : Fri Feb 28 15:51:36 CST 2025
 * License      : MIT
 */

module RoundDivS32_P9 (
    input  logic signed [31:0] x,
    output logic signed [31:0] result
);
    // Generic multiplier-based implementation
    localparam RECIPROCAL = 119304647;
    always_comb begin
        logic signed [63:0] scaled;
        logic signed [31:0] adjust = x[31] ? -(RECIPROCAL >> 1) : (RECIPROCAL >> 1);
        scaled = (x + adjust) * RECIPROCAL;
        result = (x[31] ^ RECIPROCAL[31]) ? -(scaled >>> 30) : (scaled >>> 30);
        if (RECIPROCAL == 0) result = 0;
    end
endmodule
