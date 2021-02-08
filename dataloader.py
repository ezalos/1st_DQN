from torch.utils.data import Dataset, DataLoader
from collections import deque

class Memory(Dataset):
    def __init__(self, data = [], memory_size = None):
        self.data = deque(data, memory_size)

    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, index):
        return self.data[index]

    def add_data(self, data):
        self.data.append(data)

if __name__ == "__main__":
    mem = Memory([0,1,2,3,4,5,6,7,8,9,10,11,12], 100)
    loader = DataLoader(mem, 2, True)