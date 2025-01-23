
path_to_imagenet_npz="/VIRTUAL_imagenet256_labeled.npz"
path_to_own_npz=""

python3 evaluator.py \
        $path_to_imagenet_npz \
        $path_to_own_npz