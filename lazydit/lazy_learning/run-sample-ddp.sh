export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_ckpt_path="DiT-pretrained-ckpt/DiT-XL-2-256x256.pt"
model_type="DiT-XL/2"
image_size=256
# $model_ckpt_path="DiT-pretrained-ckpt/DiT-XL-2-512x512.pt"
# model_type="DiT-XL/2"
# image_size=512
#model_ckpt_path="DiT_Llama_3B_patch2.pth"
#model_type="DiT_Llama_3B_patch2"
#image_size=256
#model_ckpt_path="DiT_Llama_7B_patch2.pth"
#model_type="DiT_Llama_7B_patch2"
#image_size=512

#model_ckpt_path="DiT-pretrained-ckpt/DiT-L-2-256x256.pt"
#model_type="DiT-L/2"
#image_size=256

output_dir="sample-ddp-outputs/"
local_batch_size=64

sample_method="ddim"
num_steps=50
#sample_method="p-sample"
#num_steps=250

torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
        --model $model_type \
        --image-size $image_size \
        --ckpt $model_ckpt_path \
        --per-proc-batch-size $local_batch_size \
        --sample-dir $output_dir \
        --sample-method $sample_method \
        --num-sampling-steps $num_steps \
        --num-fid-samples 50000 \

