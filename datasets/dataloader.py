from torch.utils.data import DataLoader

from datasets.CMUDataset import CMUData
from datasets.IEMODataset import IEMOData
from datasets.FOODDataset import FoodDataset
from datasets.BratsDataset import BraTSData
from datasets.CREMADDataset import load_cremad

class opt:
    cvNo = 1
    A_type = "comparE"
    V_type = "denseface"
    L_type = "bert_large"
    norm_method = 'trn'
    in_mem = False


def getdataloader(dataset, batch_size, data_path):
    if dataset == 'mosi':
        data = {
            'train': CMUData(data_path, 'train'),
            'valid': CMUData(data_path, 'valid'),
            'test': CMUData(data_path, 'test'),
        }
        orig_dim = data['test'].get_dim()
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }
    elif dataset == 'iemo':
        data = {
            'train': IEMOData(opt, data_path, set_name='trn'),
            'valid': IEMOData(opt, data_path, set_name='val'),
            'test': IEMOData(opt, data_path, set_name='tst'),
        }
        orig_dim = data['test'].get_dim()
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           drop_last=False,
                           collate_fn=data['test'].collate_fn)
            for ds in data.keys()
        }
    elif dataset == 'food':
        data = {
            'train': FoodDataset("data/food/features/", "data/food/texts/train_titles.csv"),
            'test': FoodDataset("data/food/features/", "data/food/texts/test_titles.csv"),
        }
        orig_dim = None
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8, 
                           shuffle=True if ds == 'train' else False)
            for ds in data.keys()
        }
    elif dataset == 'brats':
        data = {
            'train': BraTSData(root=data_path, mode='train'),
            'valid': BraTSData(root=data_path, mode='valid'),
            'test': BraTSData(root=data_path, mode='test'),
        }
        orig_dim = None
        dataLoader = {
            ds: DataLoader(data[ds],
                           batch_size=batch_size,
                           num_workers=8)
            for ds in data.keys()
        }
    elif dataset == "cremad":
        train_dataset, test_dataset = load_cremad('data/')
        dataLoader = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8),
            'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
        }
        orig_dim = None
    return dataLoader, orig_dim

if __name__ == "__main__":
    dataLoader, orig_dim = getdataloader('cremad', 32, 'data/')
    print(len(dataLoader['train']))
    print(len(dataLoader['test']))
    print(orig_dim)