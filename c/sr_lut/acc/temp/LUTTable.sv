/**
 * Module       : LUTTable
 * Input        : x[7:0] - Signed/Unsigned 8-bit interger
 * Output       : y[7:0] - Unsigned/Signed 8-bit interger
 * Description  : LUT Table for read and write.
 * Author       : lancerstadium
 * Date         : ${date}
 * License      : MIT
 */
<%
    import math
    import numpy as np
    
    # 计算总元素个数并展平
    index_size = shape[0]
    total_size = np.prod(shape)
    flattened_shape = (total_size,)
    
    # 计算地址位宽
    def clog2(x):
        return math.ceil(math.log2(x)) if x >0 else 0
    
    # 生成可读性强的初始化内容（每行batch个元素）
    def init_content(table, original_shape):
        flat_data = table.flatten()
        lines = []
        current_line = []
        
        # 添加形状注释
        lines.append(f"// Original Shape: {original_shape}, Total Entries: {len(flat_data)}")
        lines.append("'{")
        
        for idx, val in enumerate(flat_data):
            # 生成多维索引注释（每行第一个元素）
            if idx % BATCH_LEN == 0:
                multi_index = np.unravel_index(idx, original_shape)
                index_comment = f"/* {multi_index} */"
                lines.append("    " + index_comment + "\t")
            
            # 数值格式化
            hex_val = f"8'sh{val & 0xFF:02x}"
            current_line.append(hex_val)
            
            # 每batch个元素
            if len(current_line) >= BATCH_LEN and not idx == len(flat_data) - 1:
                lines[-1] += ("    " + ", ".join(current_line) + ", ")
                current_line = []
        
        # 处理最后一行
        if current_line:
            lines[-1] += ("    " + ", ".join(current_line))
        
        lines.append("};")
        return '\n'.join(lines)
%>
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off CASEINCOMPLETE */
module ${module_name} (
    // 系统接口
    input  logic         clk,      // 全局时钟
    input  logic         rst_n,    // 异步复位
    // 批量读取控制
    input  logic         burst_start,  // 脉冲启动批量读取
    output logic         burst_done,   // 批量读取完成标志
    input  logic [${clog2(index_size)-1}:0]    base_addr,  // 非展平基地址
    // 数据输出
    output logic signed [7:0] data [0:${BATCH_LEN-1}]  // 并行输出
);

    (* rom_style = "block" *) 
    logic signed [7:0] lut_mem [0:${total_size-1}]
    % if 'table' in locals():
    =
    ${init_content(table, shape)}
    % else:
    ;
    % endif

    // 状态机增强版（防止锁死）
    typedef enum logic [1:0] {
        IDLE,
        ACTIVE,
        ERROR_STATE
    } burst_state_t;
    
    burst_state_t state;
    logic [${clog2(BATCH_LEN)+1}:0] counter;  // 增加安全位宽
    logic [${clog2(total_size)-1}:0] addr_reg [0:${BATCH_LEN-1}];

    // 地址有效性检查
    logic addr_valid;
    assign addr_valid = (base_addr * ${BATCH_LEN} + ${BATCH_LEN}) <= ${total_size};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            counter <= 0;
            burst_done <= 0;
            foreach(addr_reg[i]) addr_reg[i] <= 0;
        end else case (state)
            IDLE: begin
                burst_done <= 0;
                if (burst_start && addr_valid) begin
                    state <= ACTIVE;
                    counter <= ${BATCH_LEN};
                    foreach(addr_reg[i]) 
                        addr_reg[i] <= base_addr * ${BATCH_LEN} + i;
                end else if (burst_start) begin
                    state <= ERROR_STATE;  // 处理非法地址
                end
            end
            
            ACTIVE: begin
                counter <= counter - 1;
                foreach(addr_reg[i])
                    addr_reg[i] <= addr_reg[i] + ${BATCH_LEN};
                
                if (counter == 1) begin
                    state <= IDLE;
                    burst_done <= 1;
                end
            end
            
            ERROR_STATE: begin
                // 可扩展错误恢复逻辑
                state <= IDLE;
            end
        endcase
    end

    generate
        for (genvar i = 0; i < ${BATCH_LEN}; i++) begin : pipeline
            always_ff @(posedge clk) begin
                if (state == ACTIVE) begin
                    data[i] <= lut_mem[addr_reg[i]];
                end else begin
                    data[i] <= '0;  // 非活动状态清零
                end
            end
        end
    endgenerate

endmodule
/* verilator lint_on CASEINCOMPLETE */
/* verilator lint_on WIDTHTRUNC */
