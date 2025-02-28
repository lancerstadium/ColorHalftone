OBJ_DIR		:= obj_dir
CORE  		:= Acc
HWDIR		:= rtl
SWDIR		:= sim
VTOP		:= $(HWDIR)/$(CORE)Top.sv
CTOP		:= $(SWDIR)/V$(CORE)Top.cpp
CSIM		:= V$(CORE)Top
WAVE  		:= $(OBJ_DIR)/wave.vcd
VSRC		:= $(filter-out $(VTOP) %TB.sv, $(wildcard $(HWDIR)/*.sv))
CSRC		:= $(wildcard $(SWDIR)/*.c $(SWDIR)/*.cc $(SWDIR)/*.cpp)

.PHONY: all sim run wave clean

all: sim

sim:
	@echo "Building simulation..."
	@verilator --cc --exe --build -Wall --trace --top-module $(CORE)Top $(VTOP) $(VSRC) $(CTOP)

run: $(OBJ_DIR)/$(CSIM)
	@cd $(OBJ_DIR) && ./$(CSIM)

wave: $(WAVE)
	@gtkwave $(WAVE)

clean:
	rm -rf $(OBJ_DIR)