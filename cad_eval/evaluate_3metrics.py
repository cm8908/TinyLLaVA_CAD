from glob import glob
import pickle
from tqdm import tqdm
import numpy as np
import argparse
import sys, os
import multiprocessing as mp
sys.path.append(".."); sys.path.append(".")
from pythonocc_operator.lib.visualize import CADsolid2pc
from pythonocc_operator.lib.macro import *
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file
from pythonocc_operator.py2step import code2shape

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--analyze_brep', action='store_true')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--train_dir', type=str, default=None, required=True)
# parser.add_argument('--compare_shape', action='store_true')
# parser.add_argument('--novel', action='store_true')
# parser.add_argument('--unique', action='store_true')
args = parser.parse_args()

result_dir = args.src

filenames = sorted(os.listdir(result_dir))


def is_valid(path) -> bool:
    """
    Returns:
        valid_sample: bool
        valid_shapes: TopoDS_Shape | None
        data_id: int
    """
    data_id = os.path.basename(path).split('.')[0]
    try:
        shape = read_step_file(path)
    except Exception as e:
        print('read step failed', path)
        return False, None, data_id
    
    print(f'Processing {data_id}...')
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print('convert pc failed', data_id)
        return False, None, data_id
    
    if args.analyze_brep:
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            print("validity check failed", data_id)
            return False, None, data_id
    
    return True, shape, data_id # valid(bool), valid_shape, data_id


def read_target_files(fname):
    try:
        shape = read_step_file(fname)
        return shape
    except:
        print(f'[TrainDir] Failed to read {fname}')
        return None

                    
def is_novel(sample) -> bool:
    for target in total_tgt_list:
        if sample.IsPartner(target):
            return False
    return True

def is_unique(sample, idx) -> bool:
    for i, target in enumerate(valid_queries):
        if idx == i:
            continue
        if sample.IsPartner(target):
            return False
    return True

novel_samples = []
unique_samples = []

if args.parallel:
    # Calculate validity 
    valid_samples, valid_shapes, data_ids = zip(*mp.Pool(processes=8).map(is_valid, tqdm([os.path.join(result_dir, name) for name in filenames])))

    valid_shapes = list(filter(None, valid_shapes))  # Eliminate None i.e. invalid shapes
    valid_queries = valid_shapes   # For uniqueness calculation

    invalid_data_ids = np.array(data_ids)[np.array(valid_samples) == False]

    # Collect train files for novelty calculation
    target_dir = args.train_dir
    filenames = glob(os.path.join(target_dir, '*.step'))
    total_tgt_list = mp.Pool(30).map(read_target_files, tqdm(filenames, desc=f'Collecting step files in {args.train_dir}'))

    # Calculate novelty 
    # novel_samples = mp.Pool(8).map(is_novel, valid_queries)  #TODO:

    # Calculate uniqueness
    unique_samples = mp.Pool(8).starmap(is_unique, tqdm([(item, i) for i, item in enumerate(valid_queries)]))
        
else:
    raise NotImplementedError
    valid_samples = []
    for name in tqdm(filenames):
        path = os.path.join(result_dir, name)
        valid_samples.append(is_valid(path))

validity = np.array(valid_samples).mean()
novelty = np.array(novel_samples).mean()
uniqueness = np.array(unique_samples).mean()
print(len(valid_shapes))
print(f'Validity: {validity}, Novelty: {novelty}, Uniqueness: {uniqueness}')
log_dir = args.log_dir if args.log_dir is not None else '.'
save_path = os.path.join(os.path.dirname(log_dir), '3metrics.txt')
with open(save_path, 'w') as f:
    log_str = f'<Validity>\nN_TOTAL={len(valid_samples)}, N_VALID={len(valid_shapes)}, Validity={validity}\n\n' + \
              f'<Novelty>\nN_TOTAL={len(novel_samples)}, N_NOVEL={np.array(novel_samples).sum()}, Novelty={novelty}\n\n' + \
              f'<Uniqueness>\nN_TOTAL={len(unique_samples)}, N_UNIQUE={np.array(unique_samples).sum()}, Uniqueness={uniqueness}\n'
    log_str += f'\nAnalyze_brep: {args.analyze_brep}\n'
    log_str += '\nInvalid data ids:\n'
    for data_id in invalid_data_ids:
        log_str += f'{data_id}\n'
    f.write(log_str)

