/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : Tue Mar  4 16:41:09 CST 2025
 * License      : MIT
 */

module ClampS32_S6 (
    input  logic signed [31:0] x,
    output logic signed [5:0] y
);
    // Signed clamping parameters
    localparam MIN_VAL = - (1 << (6 - 1));
    localparam MAX_VAL =   (1 << (6 - 1)) - 1;

    always_comb begin
        logic sign_bit = x[31];
        logic [25:0] upper_bits = x[31:6];
        logic overflow;

        // 溢出条件判断优化
        if (sign_bit) begin
            // 负数：高位必须全为1，否则需要钳位
            overflow = (upper_bits != {(26){1'b1}});
        end else begin
            // 正数：高位全0且低位最高位为0（防止截断后符号反转）
            overflow = (upper_bits != 0) || x[5];
        end

        y = overflow ? 
            (sign_bit ? MIN_VAL[5:0] : MAX_VAL[5:0]) : 
            x[5:0];
    end
endmodule
