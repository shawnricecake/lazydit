export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

imagenet_path="path_to_imagenet"

model_ckpt_path="DiT-pretrained-ckpt/DiT-XL-2-256x256.pt"
model_type="DiT-XL/2"
image_size=256
# model_ckpt_path="DiT-pretrained-ckpt/DiT-XL-2-512x512.pt"
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

#sample_method="ddim"
num_steps=50
#sample_method="p-sample"
#num_steps=250

output_path="output_dir"
epochs=1
max_step=500
local_batch_size=64
learning_rate=1e-4
attn_lazy_loss_ratio=1e-7
mlp_lazy_loss_ratio=1e-7

torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 lazy_learning_imagenet.py \
        --data-path $imagenet_path \
        --model $model_type \
        --ckpt $model_ckpt_path \
        --image-size $image_size \
        --num-sampling-steps $num_steps \
        --epochs $epochs \
        --local-batch-size $local_batch_size \
        --lr $learning_rate \
        --attn_lazy_loss_ratio $attn_lazy_loss_ratio \
        --mlp_lazy_loss_ratio $mlp_lazy_loss_ratio \
        --log-every 100 \
        --ckpt-every 100 \
        --max_step $max_step \
        --results-dir $output_path \


