/**
 * Module       : LUTTable
 * Input        : x[31:0] - Signed/Unsigned 8-bit interger
 * Output       : y[31:0] - Unsigned/Signed 8-bit interger
 * Description  : LUT Table for read and write.
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    import math
    import numpy as np
    
    # 计算地址位宽
    def clog2(x):
        return math.ceil(math.log2(x)) if x >0 else 0
    
    # 生成可读性强的初始化内容（每行batch个元素）
    def init_content(table, original_shape):
        flat_data = table.flatten()
        lines = []
        current_line = []
        lines.append("'{")
        hex_val = []
        
        for idx, val in enumerate(flat_data):
            # 生成多维索引注释（每行第一个元素）
            if idx % (BATCH_LEN * 4) == 0:
                multi_index = np.unravel_index(idx, original_shape)
                index_comment = f"/* {multi_index} */"
                lines.append("    " + index_comment + "\t")
            
            # 数值格式化: 转换为 uint8
            hex_val.append(val & 0xFF)

            if idx % BATCH_LEN == 3:
                # 连接四个数为32bit
                hex_val_32bit = int.from_bytes(hex_val, byteorder='little', signed=False)
                hex_val_32bit_str = "32'h{:08X}".format(hex_val_32bit)
                current_line.append(hex_val_32bit_str)
                hex_val = []
            
            # 每batch个元素
            if len(current_line) >= BATCH_LEN and not idx == len(flat_data) - 1:
                lines[-1] += ("    " + ", ".join(current_line) + ", ")
                current_line = []
        
        # 处理最后一行
        if current_line:
            lines[-1] += ("    " + ", ".join(current_line))
        
        lines.append("};")
        return '\n'.join(lines)

    BATCH_LEN = context.get('BATCH_LEN', 4)
    # 计算总元素个数并展平
    index_size = shape[0]
    total_size = (np.prod(shape) // 4)
    flattened_shape = (total_size,)
%>
module ${module_name} (
    input  logic        en_write,                           // 脉冲启动批量写入
    input  logic        en_read,                            // 脉冲启动批量读取
    input  logic        [${clog2(index_size)-1}:0] base,    // 非展平基地址
    input  logic        [31:0] wdata [0:${BATCH_LEN-1}],    // 并行输入
    output logic        [31:0] rdata [0:${BATCH_LEN-1}]     // 并行输出
);

    // Original Shape <int8x4>: (${index_size}, ${BATCH_LEN}), Total Entries: ${index_size}
    logic [31:0] lut_mem [0:${total_size-1}]
    % if 'table' in locals() and not is_dynamic:
    =
    ${init_content(table, shape)}
    % else:
    ;
    % endif

    // 写入数据
    always_latch begin
        if (en_write) begin
            % for i in range(BATCH_LEN):
            lut_mem[base * ${BATCH_LEN} + ${i}] = wdata[${i}];
            % endfor
        end else begin
            ;
        end
    end

    // 读取数据
    always_latch begin
        if (en_read) begin
            % for i in range(BATCH_LEN):
            rdata[${i}] = lut_mem[base * ${BATCH_LEN} + ${i}];
            % endfor
        end else begin
            ;
        end
    end

endmodule
