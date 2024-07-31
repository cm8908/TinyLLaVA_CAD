#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH="/home/jung/TinyLLaVA_Factory/outputs/OpenECAD-Gemma-SigLip-2.4B-lora-split"
MODEL_NAME="OpenECAD-Gemma-SigLip-2.4B-lora-split"
EVAL_IMG_DIR="/home/jung/TinyLLaVA_Factory/dataset/captioncad/images"
EVAL_TEXT_PATH="dataset/captioncad/captioncad_1552.jsonl"
MAX_TOKENS=3072

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_TEXT_PATH \
        --image-folder $EVAL_IMG_DIR \
        --answers-file $MODEL_PATH/generated_captioncad/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --max_new_tokens $MAX_TOKENS \
        --conv-mode gemma &
done

wait

output_file=$MODEL_PATH/generated_captioncad/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $MODEL_PATH/generated_captioncad/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
