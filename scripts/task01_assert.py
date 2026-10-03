import toploc
import torch

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
print("gpu", torch.cuda.get_device_name(0))
print("torch", torch.__version__)
print("toploc_file", toploc.__file__)
print("toploc_attrs", [a for a in dir(toploc) if not a.startswith("_")])
