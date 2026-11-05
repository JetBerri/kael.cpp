import kael
from kael import _C


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if not tensors or any(t.shape[0] != tensors[0].shape[0] for t in tensors):
            raise ValueError(
                "TensorDataset needs tensors with the same first dimension"
            )
        self.tensors = tensors

    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        if batch_size <= 0:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        size = len(self.dataset)
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    def _order(self):
        size = len(self.dataset)
        return kael.randperm(size) if self.shuffle else kael.arange(size)

    def __iter__(self):
        order = self._order()
        for start in range(0, len(self.dataset), self.batch_size):
            batch = order[start : start + self.batch_size]
            if self.drop_last and batch.numel() < self.batch_size:
                return
            if isinstance(self.dataset, TensorDataset):
                yield tuple(
                    kael.index_select(t, 0, batch) for t in self.dataset.tensors
                )
            else:
                yield self._collate([self.dataset[int(i)] for i in batch.tolist()])

    @staticmethod
    def _collate(samples):
        if isinstance(samples[0], tuple):
            return tuple(kael.stack(list(column)) for column in zip(*samples))
        return kael.stack(samples)


def load_mnist(directory="data/mnist", train=True):
    return _C._load_mnist(str(directory), train)
