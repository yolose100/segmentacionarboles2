import torch
import gc
for obj in gc.get_objects():
	if torch.is_tensor(obj) or (hasattr(obj, 'data')):
		print(type(obj), obj.size())
