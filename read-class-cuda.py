from typing import Dict
import numpy as np
import torch
import ray

class TorchPredictor:

    def __init__(self):
        self.model = torch.nn.Identity().cuda()
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = torch.as_tensor(batch["data"], dtype=torch.float32).cuda()
        with torch.inference_mode():
            batch["output"] = self.model(inputs).detach().cpu().numpy()
        return batch

ds = (
    ray.data.from_numpy(np.ones((32, 100)))
    .map_batches(
        TorchPredictor,
        # workers with one GPU each
        concurrency=1,
        # Batch size is required if you're using GPUs.
        batch_size=4,
        num_gpus=1
    )
)
ds.show()
