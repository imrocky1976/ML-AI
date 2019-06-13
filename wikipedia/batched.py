import numpy as np


def batched(iterator, batch_size):
    """Group a numerical stream into batches and yield them as Numpy arrays."""
    while True:
        data = np.zeros(batch_size, dtype=np.int32)
        target = np.zeros(batch_size, dtype=np.int32)
        for index in range(batch_size):
            data[index], target[index] = next(iterator)
        yield data, target