/**
 * Module       : RoundDivS32
 * Input        : x[31:0] - Signed 32-bit dividend
 *              : y[31:0] - Signed 32-bit divisor
 * Output       : result[31:0] - Signed 32-bit quotient
 * Description  : Signed 32-bit division with rounding to nearest integer.
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    import math

    dynamic_y = context.get('dynamic_y', False)
    y_val = context.get('y_val', 4)
    x_sign_positive = context.get('x_sign_positive', True)
    module_name = "RoundDivS32" if dynamic_y else f"RoundDivS32_{'P' if x_sign_positive else 'N'}{y_val}"
    
    is_power_of_two = not dynamic_y and (y_val & (y_val - 1)) == 0
    if is_power_of_two:
        shift_bits = int(math.log2(y_val))
        reciprocal_val = 0 
    else:
        reciprocal_val = (1 << 30)
%>
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
module ${module_name} (
    input  logic signed [31:0] x,
    % if dynamic_y:
    input  logic signed [31:0] y,  // 动态除数接口
    % endif
    output logic signed [31:0] result
);
    % if is_power_of_two:
    // 静态2^n优化路径
    localparam SHIFT = ${shift_bits};
    always_comb begin
        logic signed [31:0] adj = ${f'{y_val} >> 1' if not dynamic_y else 'y >> 1'};
        if (${'y' if dynamic_y else y_val} == 0) begin
            result = 0;
        end else begin
            // 符号统一处理
            logic signed [32:0] adjusted = x + (${'adj' if x_sign_positive else '-adj'});
            result = adjusted >>> SHIFT;
        end
    end
    % else:
    // 通用倒数乘法实现
    localparam RECIPROCAL = ${reciprocal_val};
    always_comb begin
        logic signed [63:0] scaled;
        logic signed [31:0] base_adj = ${f'{y_val} >> 1' if not dynamic_y else 'y >> 1'};
        logic signed [31:0] adj = ${'base_adj' if x_sign_positive else '-base_adj'};
        
        if (${'y' if dynamic_y else y_val} == 0) begin
            result = 0;
        end else begin
            scaled = (x + adj) * ${'RECIPROCAL' if not dynamic_y else '((1 << 30) / y)'};
            // 符号校正与精度对齐
            result = (x[31] ^ ${'y[31]' if dynamic_y else (1 if y_val < 0 else 0)}) ? 
                    -(scaled[61:30]) : scaled[61:30];
        end
    end
    % endif
endmodule
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on WIDTHEXPAND */
