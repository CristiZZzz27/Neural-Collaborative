import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import MovieLens,Download
from model.MLP import MLP
from model.GMF import GMF
from model.NeuMF import NeuMF
from train import Train
from evaluation import metrics
import os
import numpy as np
import time
from zipfile import ZipFile
from torch.utils.data import Dataset
# from parser import args

class MovieLens(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 ng_ratio:int
                 )->None:
        '''
        :param root: dir for download and train,test.
        :param df: parsed dataframe
        :param file_size: large of small. if size if large then it will load(download) 20M dataset. if small then, it will load(download) 100K dataset.
        :param download: if true, it will down load from url.
        '''
        super(MovieLens, self).__init__()

        self.df = df
        self.total_df = total_df
        self.ng_ratio = ng_ratio

        # self._data_label_split()
        self.users, self.items, self.labels = self._negative_sampling()



    def __len__(self) -> int:
        '''
        get lenght of data
        :return: len(data)
        '''
        return len(self.users)


    def __getitem__(self, index):
        '''
        transform userId[index], item[inedx] to Tensor.
        and return to Datalaoder object.
        :param index: idex for dataset.
        :return: user,item,rating
        '''
        return self.users[index], self.items[index], self.labels[index]


    def _negative_sampling(self) :
        '''
        sampling one positive feedback per #(ng ratio) negative feedback
        :return: list of user, list of item,list of target
        '''
        df = self.df
        total_df = self.total_df
        users, items, labels = [], [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
        all_movieIds = total_df['movieId'].unique()
        # negative feedback dataset ratio
        negative_ratio = self.ng_ratio
        for u, i in user_item_set:
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1.0)

            #visited check
            visited=[]
            visited.append(i)
            # negative instance
            for i in range(negative_ratio):
                # first item random choice
                negative_item = np.random.choice(all_movieIds)
                # check if item and user has interaction, if true then set new value from random

                while (u, negative_item) in total_user_item_set or negative_item in visited :
                    negative_item = np.random.choice(all_movieIds)
                users.append(u)
                items.append(negative_item)
                visited.append(negative_item)
                labels.append(0.0)
        print(f"negative sampled data: {len(labels)}")
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

if not os.path.exists('..\\NCF-master\\resource\\ml-1m'):
    with ZipFile('..\\NCF-master\\resource\\ml-1m.zip', "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path='..\\NCF-master\\resource')
        print("Downloading Complete!")

def _read_ratings_csv(fname) -> pd.DataFrame:
    '''
    at first, check if file exists. if it doesn't then call _download().
    it will read ratings.csv, and transform to dataframe.
    it will drop columns=['timestamp'].
    :return:
    '''
    print("Reading file")

    df = pd.read_csv(fname, sep="::", header=None,
                        names=['userId', 'movieId', 'ratings', 'timestamp'])
    df = df.drop(columns=['timestamp'])
    print("Reading Complete!")
    return df

def split_train_test(df) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    pick each unique userid row, and add to the testset, delete from trainset.
    :return: (pd.DataFrame,pd.DataFrame,pd.DataFrame)
    '''
    train_dataframe = df
    test_dataframe = df.sample(frac=1).drop_duplicates(['userId'])
    tmp_dataframe = pd.concat([train_dataframe, test_dataframe])
    train_dataframe = tmp_dataframe.drop_duplicates(keep=False)

    # explicit feedback -> implicit feedback
    # ignore warnings
    np.warnings.filterwarnings('ignore')
    train_dataframe.loc[:, 'rating'] = 1
    test_dataframe.loc[:, 'rating'] = 1

    print(f"len(total): {len(df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
    return df, train_dataframe, test_dataframe,

fname=os.path.join('..\\NCF-master\\resource\\ml-1m', 'ratings.dat')
df=_read_ratings_csv(fname)
total_dataframe, train_dataframe, test_dataframe = split_train_test(df)

if 'zip' in dir():
	del(zip)

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())


# make torch.utils.data.Data object
train_set = MovieLens(df=train_dataframe,total_df=total_dataframe,ng_ratio=4)
test_set = MovieLens(df=test_dataframe,total_df=total_dataframe,ng_ratio=99)


loaded_model = torch.load("..\\NCF-master\\pretrain\\NeuMF.pth")
loaded_model.cuda()

dataloader_test = DataLoader(dataset=test_set,
                             batch_size=100,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True
                             )

criterion = torch.nn.BCELoss()
for user,item,target in dataloader_test:
    user,item,target=user.to(device),item.to(device),target.float().to(device)
    pred = loaded_model(user, item)
    cost = criterion(pred,target)

HR, NDCG = metrics(loaded_model,dataloader_test,10,device)
print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))