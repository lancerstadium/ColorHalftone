# https://docs.makotemplates.org/en/latest/
from mako.template import Template
import numpy as np
import os


class RTLGenerator:
    def __init__(self, out_dir='./rtl', temp_dir='./temp'):
        self.template = None
        self.temp_dir = temp_dir
        self.out_dir = out_dir
        self.suffix = '.sv'

    def gen(self, out_name, temp_path, **kwargs):
        '''Write RTL to out dir'''
        self.template = Template(filename=temp_path)
        rtl = self.template.render(**kwargs)
        with open(os.path.join(self.out_dir, out_name + self.suffix), 'w') as f:
            f.write(rtl)
        print(f'Generate {out_name}{self.suffix} successfully!')

    def CastD8(self, is_s2u=False):
        '''Generate CastD8'''
        self.gen(
            out_name='CastD8_U2S' if not is_s2u else 'CastD8_S2U',
            temp_path=os.path.join(self.temp_dir, 'CastD8.sv'),
            date=os.popen('date').read().strip(),
            is_s2u=is_s2u
        )

    def RoundDivS32(self, divisor=None):
        '''Generate RoundDivS32'''
        is_dynamic = divisor is None
        is_power_of_two = False
        shift_bits = 0
        reciprocal = 0
        x_sign_positive = (divisor > 0) if divisor else True
        
        if not is_dynamic:
            y_abs = abs(divisor)
            if y_abs & (y_abs - 1) == 0 and y_abs != 0:
                is_power_of_two = True
                shift_bits = (y_abs.bit_length() - 1)
            
            # 计算固定倒数 (1<<30)/y_abs
            reciprocal = (1 << 30) // y_abs if y_abs !=0 else 0

        self.gen(
            out_name='RoundDivS32' if is_dynamic else f'RoundDivS32_{"P" if divisor > 0 else "N"}{divisor}',
            temp_path=os.path.join(self.temp_dir, 'RoundDivS32.sv'),
            date=os.popen('date').read().strip(),
            y_val=divisor or 0,
            dynamic_y=is_dynamic,
            is_power_of_two=is_power_of_two,
            shift_bits=shift_bits,
            reciprocal_val=reciprocal,
            x_sign_positive=x_sign_positive
        )
        

    def ClampS32(self, bitwidth=8, sign=False):
        '''Generate ClampS32'''
        self.gen(
            out_name=f"ClampS32_{'S' if sign else 'U'}{bitwidth}",
            temp_path=os.path.join(self.temp_dir, 'ClampS32.sv'),
            date=os.popen('date').read().strip(),
            bitwidth=bitwidth,
            sign=sign
        )

    def DepthLUT(self, C=3, H=50, W=50, upscale=4, ksz=3, datawidth=8, msb_path="", lsb_path=""):
        msb_name = msb_path.split('/')[-1].split('.')[0]
        lsb_name = lsb_path.split('/')[-1].split('.')[0]
        self.LUTTable(npy_path=msb_path, batch_len=upscale**2, npy_name=msb_name)
        self.LUTTable(npy_path=lsb_path, batch_len=upscale**2, npy_name=lsb_name)
        self.gen(
            out_name=f"DepthLUT_{C}x{H}x{W}_K{ksz}_U{upscale}_D{datawidth}",
            temp_path=os.path.join(self.temp_dir, 'DepthLUT.sv'),
            date=os.popen('date').read().strip(),
            C=C,
            H=H,
            W=W,
            UPSCALE=upscale,
            KSZ=ksz,
            DW=datawidth,
            msb_name=msb_name,
            lsb_name=lsb_name
        )

    def LUTTable(self, npy_path='lut.npy', batch_len=16, npy_name='lut'):
        '''Generate LUTTable'''
        table = np.load(npy_path).astype(np.int8)
        # squeeze table (delete 1-dim)
        table = np.squeeze(table)
        shape = table.shape
        self.gen(
            out_name=f'LUTTable_{npy_name}',
            temp_path=os.path.join(self.temp_dir, 'LUTTable.sv'),
            module_name=f'LUTTable_{npy_name}',
            date=os.popen('date').read().strip(),
            table=table,
            shape=shape,
            BATCH_LEN=batch_len
        )
        


        
 
if __name__ == "__main__":
    gen = RTLGenerator()
    gen.CastD8(True)
    gen.CastD8(False)
    gen.RoundDivS32(9)
    gen.RoundDivS32(16)
    gen.ClampS32(6, True)
    gen.ClampS32(2, False)
    gen.DepthLUT(msb_path='../../../lut/TinyLUT/x4_4b_i8_s1_D_H6.npy', lsb_path='../../../lut/TinyLUT/x4_4b_i8_s1_D_L2.npy')