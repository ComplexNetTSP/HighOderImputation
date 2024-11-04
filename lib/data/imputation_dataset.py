import torch
import numpy as np

from lib.datasets import AirQuality
from lib.data import TemporalDataset, SpatioTemporalDataset


class ImputationDataset(TemporalDataset):

    def __init__(self, data,
                 index=None,
                 mask=None,
                 eval_mask=None,
                 freq=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 stride=1,
                 exogenous=None):
        if mask is None:
            mask = np.ones_like(data)
        if exogenous is None:
            exogenous = dict()
        exogenous['mask_window'] = mask
        if eval_mask is not None:
            exogenous['eval_mask_window'] = eval_mask
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                exogenous=exogenous,
                                                trend=trend,
                                                scaler=scaler,
                                                freq=freq,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride)

    def get(self, item, preprocess=False):
        res, transform = super(ImputationDataset, self).get(item, preprocess)
        res['x'] = torch.where(res['mask'], res['x'], torch.zeros_like(res['x']))
        return res, transform


class GraphImputationDataset(ImputationDataset, SpatioTemporalDataset):
    pass


if __name__ == '__main__':
    AQ = AirQuality()

    #print(*AQ.numpy(return_index=True))
    dataset_cls = ImputationDataset
    torch_dataset = dataset_cls(*AQ.numpy(return_idx=True),
                                window=24,
                                stride=1)

    print(torch_dataset)
    #print(AQ.eval_mask.shape)

    train_idxs, val_idxs, test_idxs = AQ.splitter(torch_dataset)

    adj = AQ.get_similarity(thr = 0.1)

    print(len(train_idxs))
    print(len(val_idxs))
    print(len(test_idxs))
    print(adj.shape)
    print(adj)
    #data_cls = ImputationDataset()
