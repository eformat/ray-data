from typing import Dict
import ray
import numpy as np

def map_fn_with_large_output(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    for i in range(3):
        yield {"large_output": np.ones((100, 1000))}

ds = (
    ray.data.from_items([1])
    .map_batches(map_fn_with_large_output)
)
ds.show()
