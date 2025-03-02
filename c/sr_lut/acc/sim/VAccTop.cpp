#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "verilated_vcd_c.h" // 生成vcd文件使用
#include "VAccTop.h"
#include "verilated.h"

int main (int argc, char **argv) {
    if (false && argc && argv) {}
    const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
    std::unique_ptr<VAccTop> top{new VAccTop{contextp.get()}};
    contextp->commandArgs(argc, argv);
    contextp->traceEverOn(true); // 生成波形文件使用，打开追踪功能
    VerilatedVcdC* ftp = new VerilatedVcdC; // vcd对象指针
    top->trace(ftp, 0); // 0层
    ftp->open("wave.vcd"); //设置输出的文件wave.vcd

    int flag = 0;

    top->clk =   0;
    top->rst_n = 0;
    top->a     = 0;
    

    while (!contextp->gotFinish() && ++flag < 20) {
        // int a = rand() % 160 - 80;
        // top->a = a;
        top->clk = !top->clk;
        top->a = (flag == 5);
        top->eval();

        // printf("a = %d, b = %d\n", a, top->b);
        // printf("a = %d, b = %d\n", a, ((int8_t)((u_int8_t)top->b << 2) >> 2));        // Clamp s6
        // printf("a = %d, b = %d\n", (int8_t)(a & 0xFF), top->b - 128);    // Cast s2u
        // printf("a = %d, b = %d\n", a, (int8_t)(top->b & 0xFF) + 128);    // Cast u2s
        // assert(top->f == (a ^ b));

        if(top->b == 1) {
            printf("burst done\n");
            // read data array
            int8_t *data = (int8_t *)top->data;
            for(int i = 0; i < 16; i++) {
                printf("%d,", data[i]);
            }
            break;
        }

        contextp->timeInc(1); // 时间+1，推动仿真时间
        // top->clk = !top->clk;
 
        ftp->dump(contextp->time()); // dump wave
    }

    top->final();

    ftp->close(); // 必须有

    return 0;
}