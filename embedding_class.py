import torch
from torch.utils.data import Dataset
import numpy as np

class EmbeddingSet(Dataset):
    """Dataset class compatible with Pytorch Dataset class
    """

    def __init__(self, data, labels):
        """
        data (numpy -float64): M x N numpy array that stores N dimensional representations
                        of M samples
        labels (numpy -int64): M dimensional numpy array that stores label indices
        """

        if data.shape[0] != labels.shape[0]:
            raise ValueError("Size mismatch between inputs and outputs!")

        self.x = torch.from_numpy(data)
        self.x = self.x.float()

        self.y = torch.from_numpy(labels)
        self.y = self.y.type(torch.LongTensor)
        self.length = data.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]



class EmbeddingSetAtt(Dataset):
    """Dataset class compatible with Pytorch Dataset class
    """

    def __init__(self, data, labels):
        """
        data (numpy -float64): 3 x M x N numpy array that stores N dimensional representations
                        of M samples
        labels (numpy -int64): M dimensional numpy array that stores label indices
        """

        if data.shape[1] != labels.shape[0]:
            raise ValueError("Size mismatch between inputs and outputs!")

        self.x = torch.from_numpy(data)
        self.x = self.x.float()

        self.y = torch.from_numpy(labels)
        self.y = self.y.type(torch.LongTensor)
        self.length = labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


# if __name__ == '__main__':

#     M = 64
#     N = 31
#     a = np.random.rand(M,N)
#     b = np.random.randint(5, size = M)

#     set = EmbeddingSet(a,b)
#     print(set[0])
#     print(set[1])