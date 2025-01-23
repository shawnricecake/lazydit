import torch

import sys; sys.path.append("../../")
# from lazydit.models.models_lazy_learning import DiT_models
from lazydit.models.models import DiT_models

# model 1
model = DiT_models["DiT-XL/2"](
    input_size=256 // 8,
    num_classes=1000,
)
onnx_file_path = "DiT_XL_2.onnx"

# model 2
# model = DiT_models["DiT-L/2"](
#     input_size=256 // 8,
#     num_classes=1000,
# )
# onnx_file_path = "DiT_L_2.onnx"

# model 3
# model = DiT_models["DiT-B/2"](
#     input_size=256 // 8,
#     num_classes=1000,
# )
# onnx_file_path = "DiT_B_2.onnx"

input1 = torch.randn(16, 4, 32, 32)
input2 = torch.randint(0, 50, (16,))
input3 = torch.randint(0, 1000, (16,))

torch.onnx.export(
    model,
    (input1, input2, input3),
    onnx_file_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input1', 'input2', 'input3'],
    output_names=['output']
)

