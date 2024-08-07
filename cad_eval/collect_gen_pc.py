import os
import glob
import numpy as np
import h5py
# from joblib import Parallel, delayed
import multiprocessing as mp
import argparse
import sys
sys.path.append(".."); sys.path.append(".")
from pythonocc_operator.lib.pc_utils import write_ply
from pythonocc_operator.lib.visualize import CADsolid2pc
from OCC.Extend.DataExchange import read_step_file


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--output', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
args = parser.parse_args()

SAVE_DIR = args.output
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def process_one(path):
    data_id = os.path.basename(path).split('.')[0]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    # print("[processing] {}".format(data_id))
    try:
        shape = read_step_file(path)
    except Exception as e:
        print('read step failed', path)
        return 

    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return 
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)


all_paths = glob.glob(os.path.join(args.src, "*.step"))
# Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in all_paths)
mp.Pool(8).map(process_one, all_paths)
