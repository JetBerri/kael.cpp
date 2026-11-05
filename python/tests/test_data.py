import pytest

import kael
from kael.data import DataLoader, Dataset, TensorDataset, load_mnist


def test_tensor_dataset():
    data = TensorDataset(kael.arange(4), kael.arange(4) * 2)
    assert len(data) == 4
    assert data[1][1].item() == 2
    with pytest.raises(ValueError):
        TensorDataset(kael.arange(4), kael.arange(3))
    with pytest.raises(ValueError):
        TensorDataset()


def test_loader_batches_in_order():
    data = TensorDataset(kael.arange(5))
    batches = [batch.tolist() for (batch,) in DataLoader(data, batch_size=2)]
    assert batches == [[0, 1], [2, 3], [4]]
    assert len(DataLoader(data, batch_size=2)) == 3


def test_loader_drop_last():
    data = TensorDataset(kael.arange(5))
    batches = [
        batch.tolist() for (batch,) in DataLoader(data, batch_size=2, drop_last=True)
    ]
    assert batches == [[0, 1], [2, 3]]
    assert len(DataLoader(data, batch_size=2, drop_last=True)) == 2


def test_loader_shuffle_covers_everything():
    kael.manual_seed(1)
    data = TensorDataset(kael.arange(50))
    seen = [
        value
        for (batch,) in DataLoader(data, batch_size=7, shuffle=True)
        for value in batch.tolist()
    ]
    assert sorted(seen) == list(range(50))
    assert seen != list(range(50))


def test_loader_with_custom_dataset():
    class Squares(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return kael.tensor(index), kael.tensor(index * index)

    batches = list(DataLoader(Squares(), batch_size=3))
    assert batches[0][1].tolist() == [0, 1, 4]


def test_abstract_dataset():
    with pytest.raises(NotImplementedError):
        len(Dataset())


def test_loader_validates_batch_size():
    with pytest.raises(ValueError):
        DataLoader(TensorDataset(kael.arange(2)), batch_size=0)


def test_missing_mnist_is_an_error():
    with pytest.raises(RuntimeError):
        load_mnist("/nonexistent")
