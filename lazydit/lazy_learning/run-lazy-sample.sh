export CUDA_VISIBLE_DEVICES=0

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

sample_method="ddim"
num_steps=50
#sample_method="p-sample"
#num_steps=250

lazy_learning_weights="path_to_lazy_learning_weights"
attn_threshold=0.5
mlp_threshold=0.5

python3 -u lazy_sample.py \
        --model $model_type \
        --image-size $image_size \
        --ckpt $model_ckpt_path \
        --lazy_learning_weights $lazy_learning_weights \
        --sample-method $sample_method \
        --num-sampling-steps $num_steps \
        --attn_threshold $attn_threshold \
        --mlp_threshold $mlp_threshold \
