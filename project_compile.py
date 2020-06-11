#!/usr/bin/python3

import os
import configparser
import subprocess


RUST_FMT_STR = '''
pub const HBAR: f32 = {hbar};
pub const M: f32 = {m};

pub const DX: f32 = {dx};
pub const DT: f32 = {dt};

pub const WIDTH: usize = {width};
pub const HEIGHT: usize = {height};

pub const NUM_FRAMES: usize = {frames};

pub const SLIT_HEIGHT: isize = {slits[height]};
pub const SLIT_1_START: isize = {slits[1][start]};
pub const SLIT_1_END: isize = {slits[1][end]};
pub const SLIT_2_START: isize = {slits[2][start]};
pub const SLIT_2_END: isize = {slits[2][end]};
pub const DISPATCH_LAYOUT : [u32 ; 3] = [{dispatch x}, {dispatch y}, 1];
pub const SHADER_LAYOUT : [usize ; 3] = [{layout x}, {layout y}, 1];
'''


def strclamp(needle, haystack):
    retvl = None
    needle = needle.lower()
    for possible in haystack:
        if possible.lower() in needle:
            assert retvl is None, 'ERROR: Found multiple matches: {} -> {} / {}'.format(
                needle, retvl, possible)
            retvl = possible.lower()
    return retvl


def idx_or_units(raw_v, delta):
    if '.' in raw_v:
        x_val = float(raw_v)
        return int(x_val/delta)
    else:
        return int(raw_v)


def get_config(fname="./config.ini"):
    config = configparser.ConfigParser()
    config.read(fname)
    retvl = {}
    for s in config.sections():
        if s.lower() == 'slits':
            continue
        d = config[s]
        for k in d:
            raw_v = d[k]
            if (nk := strclamp(k, ['frames', 'width', 'height'])) is not None:
                retvl[nk] = int(raw_v)
            elif (nk := strclamp(k, ['hbar', 'dx', 'dt'])) is not None or (nk := k.lower()) == 'm':
                retvl[nk] = float(raw_v)
            else:
                retvl[k.lower()] = raw_v
    dx = retvl['dx']
    dt = retvl['dt']
    slit_params = config['Slits']
    for k in slit_params:
        if (nk := strclamp(k, ['height'])) is not None:
            retvl.setdefault('slits', {})[
                nk] = idx_or_units(slit_params[k], dx)
            continue
        slit_n = int(k[-1])
        nk = strclamp(k, ['start', 'end'])
        if nk is not None:
            retvl.setdefault('slits', {}).setdefault(slit_n, {})[
                nk] = idx_or_units(slit_params[k], dx)
    return retvl


def compile_rust(conf, dr):
    os.chdir(dr)
    with open('src/my_consts.rs', 'w') as f:
        msg = RUST_FMT_STR.format_map(conf)
        f.write(msg)
    subprocess.check_call(
        'cargo clean && cargo build --release'.format_map(conf), shell=True
    )
    os.chdir('..')


def compile_c_helpers(conf, fl):
    C_PARAMS_FMT = '''
    -DHBAR='({hbar})'
    -DM='({m})'
    -DDX='({dx})'
    -DDT='({dt})'
    -DWIDTH='({width})'
    -DHEIGHT='({height})'
    -DNUM_FRAMES='({frames})'
    -DSLIT_HEIGHT='({slits[height]})'
    -DSLIT_1_START='({slits[1][start]})'
    -DSLIT_1_END='({slits[1][end]})'
    -DSLIT_2_START='({slits[2][start]})'
    -DSLIT_2_END='({slits[2][end]})'
    '''.replace('\n', ' ')
    cmd = 'gcc -o {} {}.c -lavcodec -lavutil -lm -std=gnu99 -Werror '.format(
        fl, fl) + C_PARAMS_FMT.format_map(conf)
    if 'debug frames' in conf:
        cmd += '-DDBG_WRITE_FRAMES '
    subprocess.check_call(cmd, shell=True)


def compile_glsl(conf, fl):
    C_PARAMS_FMT = '''
    -DHBAR='({hbar})'
    -DM='({m})'
    -DDX='({dx})'
    -DDT='({dt})'
    -DWIDTH='({width})'
    -DHEIGHT='({height})'
    -DNUM_FRAMES='({frames})'
    -DSLIT_HEIGHT='({slits[height]})'
    -DSLIT_1_START='({slits[1][start]})'
    -DSLIT_1_END='({slits[1][end]})'
    -DSLIT_2_START='({slits[2][start]})'
    -DSLIT_2_END='({slits[2][end]})'
    '''.replace('\n', ' ')
    cmd = 'glslangValidator -o {}.spv {}.comp -V110 -g -e "main"'.format(
        fl, fl) + C_PARAMS_FMT.format_map(conf)
    subprocess.check_call(cmd, shell=True)


def output_glsl_include(conf, outfl):
    HEADER_DEFS = '''
#define HBAR ({hbar})
#define M ({m})
#define DX ({dx})
#define DT ({dt})
#define WIDTH ({width})
#define HEIGHT ({height})
#define NUM_FRAMES ({frames})
#define LAYOUT_X ({layout x})
#define LAYOUT_Y ({layout y})
#define SLIT_HEIGHT ({slits[height]})
#define SLIT_1_START ({slits[1][start]})
#define SLIT_1_END ({slits[1][end]})
#define SLIT_2_START ({slits[2][start]})
#define SLIT_2_END ({slits[2][end]})
#define DISPATCH_X ({dispatch x})
#define DISPATCH_Y ({dispatch y})
    '''.format_map(conf)
    outfh = open(outfl, 'w')
    outfh.write(HEADER_DEFS)
    outfh.close()

def compile_cuda(conf, fl):
    cmd = 'nvcc -o {} {}.cu -I cudac/ -I cudac/cuda-samples/Common/'.format(fl, fl)
    subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    conf = get_config()
    print(conf)
    output_glsl_include(conf, 'vulkanrs/res/consts.h')
    compile_glsl(conf, 'vulkanrs/res/stepper')
    compile_glsl(conf, 'vulkanrs/res/summer')
    compile_rust(conf, 'vulkanrs')
    if os.path.exists('vulkanrs.out'):
        os.unlink('vulkanrs.out')
    os.link('vulkanrs/target/release/vulkanrs', 'vulkanrs.out')
    compile_c_helpers(conf, 'utils/videoencoder')
    compile_c_helpers(conf, 'utils/make_rgb')
    output_glsl_include(conf, 'cudac/consts.h')
    compile_cuda(conf, 'cudac/main')
    if os.path.exists('cudac.out'):
        os.unlink('cudac.out')
    os.link('cudac/main', 'cudac.out')
