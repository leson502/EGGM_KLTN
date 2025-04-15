from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.CMUDataset import CMUData
from datasets.IEMODataset import IEMOData
from datasets.FOODDataset import FoodDataset
from datasets.BratsDataset import BraTSData
from datasets.CREMADDataset import load_cremad
from datasets.MoseiDataset import CMUMOSIDataset
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
            'train': FoodDataset("data/food/", "data/food/texts/train_titles.csv", "google-bert/bert-base-uncased", "google/vit-base-patch16-224"),
            'test': FoodDataset("data/food/", "data/food/texts/test_titles.csv", "google-bert/bert-base-uncased", "google/vit-base-patch16-224"),
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
    elif dataset == "mosei":
        dataset = CMUMOSIDataset("data/mosei/CMUMOSEI_features_raw_2way.pkl", 
                                 "data/mosei/features/wav2vec-large-c-UTT", 
                                 "data/mosei/features/deberta-large-4-UTT", 
                                 "data/mosei/features/manet_UTT")
        trainNum = len(dataset.trainVids)
        valNum = len(dataset.valVids)
        testNum = len(dataset.testVids)
        train_idxs = list(range(0, trainNum))
        val_idxs = list(range(trainNum, trainNum+valNum))
        test_idxs = list(range(trainNum+valNum, trainNum+valNum+testNum))
            
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)
        test_sampler = SubsetRandomSampler(test_idxs)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=dataset.collate_fn)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=dataset.collate_fn)

        dataLoader = {
            'train': train_loader,
            'valid': val_loader,
            'test': test_loader
        }
        orig_dim = [512, 1024, 1024]

    return dataLoader, orig_dim

if __name__ == "__main__":
    dataLoader, orig_dim = getdataloader('cremad', 32, 'data/')
    print(len(dataLoader['train']))
    print(len(dataLoader['test']))
    print(orig_dim)