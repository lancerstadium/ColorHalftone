.PHONY: all clean

# 可执行文件的目标名称
TARGET = lut

# 源文件
SRC = intp.c data.c lut.c main.c
INC = intp.h data.h lut.h

# 生成的目标文件
OBJ = $(SRC:.c=.o)

# 编译器
CC = gcc

# 编译选项
CFLAGS = -O2

# 如果执行 make perf，开启 -pg 选项
ifeq ($(MAKECMDGOALS),perf)
CFLAGS += -pg
endif

# 链接选项
LDFLAGS = -lm

# 默认目标
all: $(TARGET)

# 生成可执行文件
$(TARGET): $(OBJ) $(INC)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

# 编译源文件为目标文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 性能分析：make clean, 开启 -pg, 过滤掉含有 stbi 开头函数的一整行
# sudo apt-get install graphviz
# pip install gprof2dot
perf: $(TARGET)
	gprof ./$(TARGET) gmon.out | grep -v " stbi" > gprof.log
	gprof ./$(TARGET) | gprof2dot | dot -Tpng -o gprof.png

# 清理目标
clean:
	rm -f $(OBJ) $(TARGET)
