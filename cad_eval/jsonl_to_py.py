import json, os, re
from tqdm import tqdm

jsonl_path = 'outputs/OpenECAD-Gemma-SigLip-2.4B-lora-split/generated_captioncad/merge.jsonl'
save_dir = 'pyfiles/generated_captioncad'

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
            
with open(jsonl_path) as f:
    for line in tqdm(f):
        data = json.loads(line)
        text = data['text']
        id = data['question_id']
        save_py(text, id)