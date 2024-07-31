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

# if args.novel:
#     total_tgt_list = []
#     if args.compare_shape:
#         target_dir = '../../datasets/cad_data/cad_shape'
#         target_filenames_dirs = sorted(os.listdir(target_dir))
#         # print(target_filenames_dirs); exit()
#         for fnm_dir in tqdm(target_filenames_dirs, desc='Collecting shape file names...'):
#             target_filenames = sorted(glob(os.path.join(target_dir, fnm_dir) + '/*.pkl'))
#             for fnm in target_filenames:
#                 data_id = fnm.split('/')[-1].split('.')[0]
#                 with open(fnm, 'rb') as f:
#                     shape = pickle.load(f)
#                     if shape is not None:
#                         total_tgt_list.append((data_id, shape))
#     else:
#         target_dir = '../../datasets/cad_data/cad_vec'
#         with open('../../datasets/cad_data/train_val_test_split.json', 'r') as f:
#             train_filename_list = json.load(f)['train']
#         for filename in tqdm(train_filename_list, desc='Collecting vectors...'):
#             fnm = os.path.join(target_dir, filename) + '.h5'
#             data_id = fnm.split('/')[-1].split('.')[0]
#             with h5py.File(fnm, 'r') as fp:
#                 vec = fp['vec'][:].astype(int)
#             total_tgt_list.append((data_id, vec))
        
                    
# def is_novel(sample) -> bool:
#     for i, (data_id, target) in enumerate(total_tgt_list):
#         if args.compare_shape and sample.IsPartner(target):
#             return False
#         if isinstance(sample, np.ndarray) and np.array_equal(sample, target):
#             return False
#     return True

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
    valid_samples, valid_shapes, data_ids = zip(*mp.Pool(processes=8).map(is_valid, tqdm([os.path.join(result_dir, name) for name in filenames])))
    # valid_samples, valid_shapes, data_ids, valid_vectors = zip(*Parallel(n_jobs=8, verbose=2)(delayed(is_valid)(os.path.join(result_dir, name)) for name in filenames))

    # print(type(valid_shapes), valid_shapes[0], len(valid_shapes))
    valid_shapes = list(filter(None, valid_shapes))  # Eliminate None i.e. invalid shapes
    # print(type(valid_vectors), type(valid_vectors[0]), len(valid_vectors))
    valid_queries = valid_shapes 

    invalid_data_ids = np.array(data_ids)[np.array(valid_samples) == False]

    # if args.novel:
    # novel_samples = mp.Pool(8).map(is_novel, valid_queries)

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
save_path = os.path.join(os.path.dirname(log_dir), '_3metrics.txt')
with open(save_path, 'w') as f:
    log_str = f'<Validity>\nN_TOTAL={len(valid_samples)}, N_VALID={len(valid_shapes)}, Validity={validity}\n\n' + \
              f'<Novelty>\nN_TOTAL={len(novel_samples)}, N_NOVEL={np.array(novel_samples).sum()}, Novelty={novelty}\n\n' + \
              f'<Uniqueness>\nN_TOTAL={len(unique_samples)}, N_UNIQUE={np.array(unique_samples).sum()}, Uniqueness={uniqueness}\n'
    log_str += f'\nAnalyze_brep: {args.analyze_brep}\n'
    log_str += '\nInvalid data ids:\n'
    for data_id in invalid_data_ids:
        log_str += f'{data_id}\n'
    f.write(log_str)

