/**
 * Module       : ClampS32
 * Input        : x[31:0] - Signed 32-bit interger
 * Output       : y[31:0] - Clamped to the range of a sub-bit signed integer
 * Description  : Signed 32-bit clamp to the range of a sub-bit signed integer.
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    sign = context.get('sign', False)
    bitwidth = context.get('bitwidth', 8)
    upper = 32 - bitwidth
    module_name = f"ClampS32_{'S' if sign else 'U'}{bitwidth}"
%>
module ${module_name} (
    input  logic signed [31:0] x,
    output logic ${'signed' if sign else ''} [${bitwidth-1}:0] y
);
% if sign:
    // Signed clamping parameters
    localparam MIN_VAL = - (1 << (${bitwidth} - 1));
    localparam MAX_VAL =   (1 << (${bitwidth} - 1)) - 1;

    always_comb begin
        logic sign_bit = x[31];
        logic [${upper-1}:0] upper_bits = x[31:${bitwidth}];
        logic overflow;

        // 溢出条件判断优化
        if (sign_bit) begin
            // 负数：高位必须全为1，否则需要钳位
            overflow = (upper_bits != {(${upper}){1'b1}});
        end else begin
            // 正数：高位全0且低位最高位为0（防止截断后符号反转）
            overflow = (upper_bits != 0) || x[${bitwidth-1}];
        end

        y = overflow ? 
            (sign_bit ? MIN_VAL[${bitwidth-1}:0] : MAX_VAL[${bitwidth-1}:0]) : 
            x[${bitwidth-1}:0];
    end
% else:
    // Unsigned clamping parameters
    localparam MAX_VAL = (1 << ${bitwidth}) - 1;
    always_comb begin
        logic sign_bit = x[31];
        logic [${upper-1}:0] upper_bits = x[31:${bitwidth}];
        
        y = sign_bit       ? ${bitwidth}'b0 :
            (|upper_bits) ? MAX_VAL[${bitwidth-1}:0] :
            x[${bitwidth-1}:0];
    end
% endif
endmodule
