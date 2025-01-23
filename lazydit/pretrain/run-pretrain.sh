
#model_type="DiT-L/2"
model_type="DiT-B/2"

data_path="path_to_imagenet"

output_dir="path_to_output"

torchrun --nnodes=1 --nproc_per_node=4 pretrain.py \
        --model $model_type \
        --data-path $data_path \
        --results-dir $output_dir \


