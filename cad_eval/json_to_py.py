import json, os, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default='outputs/OpenECAD-Gemma-SigLip-2.4B-lora-split/generated/merge.jsonl')
parser.add_argument('--save_dir', type=str, default='outputs/OpenECAD-Gemma-SigLip-2.4B-lora-split/generated/pyfiles')
args = parser.parse_args()

file_path = args.file_path
save_dir = args.save_dir

def save_py(string, id):
    save_path = os.path.join(save_dir, id+'.py')
    with open(save_path, 'w') as f:
        str_li = string.split('\n')[2:-1]
        if '`' in str_li[-1]:
            str_li = str_li[:-1]
        if not str_li[-1].endswith(')'):
            return id
        for s in str_li:  # [2:-2] from start to end of the code
            f.write(s+'\n')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(file_path) as f:
    if file_path.endswith('.jsonl'):
        for line in tqdm(f):
            data = json.loads(line)
            text = data['text']
            id = data['question_id']
            save_py(text, id)
    elif file_path.endswith('.json'):
        data = json.load(f)
        for d in data:
            text = d['answer']
            id = d['question_id']
            print(text); exit()
            save_py(text, id)