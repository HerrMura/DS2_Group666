import h5py
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][idx] , f['hr'][idx] # each of them is a 3-channel img of size (3,h,w) or (3,sh,sw)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['lr'][str(idx)][:, :] , f['hr'][str(idx)][:, :] 

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
