import os, sys
from tqdm import tqdm; sys.path.append('..'); sys.path.append('.')
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.data.text_preprocess import TextPreprocess
from tinyllava.utils.message import Message

def defaultdict_factory():
    return {'count': 0, 'similarity': 0.0, 'tokens': []}

def cosine_similarity_multi(token_groups):
    vocab_size = max(max(tokens) for tokens in token_groups) + 1
    vectors = []
    
    for tokens in token_groups:
        vector = np.zeros(vocab_size)
        for token in tokens:
            vector[token] += 1
        vectors.append(vector)
    
    similarity_matrix = cosine_similarity(vectors)
    return np.mean(similarity_matrix[np.triu_indices(len(vectors), k=1)])

def calculate_similarity(token_groups, count, adjustment=False):
    base_similarity = cosine_similarity_multi(token_groups)
    
    # Adjust similarity based on the number of token sequences
    if adjustment:
        adjustment_factor = 1 + np.log(count) / 10  # Logarithmic scaling
    else:
        adjustment_factor = 1
    
    return base_similarity * adjustment_factor

def main(args):
    if os.path.isfile(args.src):
        raise ValueError(f"{args.src} is a file, not a directory. Please input a directory.")

    token_groups = defaultdict(defaultdict_factory)
    model, tokenizer, _, _ = load_pretrained_model('Yuki-Kokomi/OpenECAD-SigLIP-2.4B')
    text_processor = TextPreprocess(tokenizer, 'gemma')

    # Read codes and calculate similarities
    for path in tqdm(os.listdir(args.src)):
        name = path.split('.')[0]
        data_id, idx = name.split('-')
        read_code = open(os.path.join(args.src, path), 'r').read()
        
        # Preprocess text (code) tokens
        msg = Message()
        msg.add_message(read_code)
        tokens = text_processor(msg.messages, mode='eval')['input_ids'].numpy()
        # embeddings = model.encode(tokens)
        
        token_groups[data_id]['count'] += 1
        token_groups[data_id]['tokens'].append(tokens)
    
    # Write log file
    log_path = os.path.join(os.path.dirname(args.src), 'multi-view-consistency-cosine_sim.txt')
    log_file = open(log_path, 'w')
    log_file.write('data_id,similarity\n')
    average_similarity = 0.0
    for data_id in tqdm(token_groups):
        if token_groups[data_id]['count'] <= 1:
            continue
        token_groups[data_id]['similarity'] = calculate_similarity(token_groups[data_id]['tokens'],
                                                                   token_groups[data_id]['count'],
                                                                   adjustment=args.adjustment)
        average_similarity += token_groups[data_id]['similarity']
        print(data_id, token_groups[data_id]['similarity'], file=log_file)
    average_similarity /= len(token_groups)
    print(f'Total number of data: {len(token_groups)}', file=log_file)
    print(f'Average similarity: {average_similarity}', file=log_file)
    log_file.close()
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='Path to source file or directory')
    parser.add_argument('--adjustment', action='store_true', help='Adjust similarity based on the number of token sequences')
    args = parser.parse_args()

    main(args)