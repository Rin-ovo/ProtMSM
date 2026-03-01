
import numpy as np
import torch
import argparse
from logzero import logger
import os
from tqdm import tqdm
from utils import get_mlb, get_pid_and_label_list, get_go_ic, get_ppi_pid2index, get_uniprot2string,get_pid_list
from torch.utils.data import Dataset,DataLoader

class SeqDataset(Dataset):
    def __init__(self, pid_list, feature_dir, true_label, max_len=4096):
        super().__init__()
        self.pids = pid_list
        self.feature_dir = feature_dir
        self.true_labels = true_label
        self.max_len = max_len

    def __getitem__(self, index):
        pid = self.pids[index]
        feature_path = os.path.join(self.feature_dir, f"{pid}.npy")
        feature = np.load(feature_path)

        L = feature.shape[0]
        if L > self.max_len:
            feature = feature[:self.max_len, :]
        feature = torch.from_numpy(feature).float()

        label_sparse_row = self.true_labels[index]
        label_dense = label_sparse_row.toarray().flatten()
        label = torch.from_numpy(label_dense).float()

        return pid, feature, label
    def __len__(self):
        return len(self.pids)


def collate_fn(batch):
    pids, features, labels = zip(*batch)
    seq_lens = [f.shape[0] for f in features]
    max_len = max(seq_lens)

    hidden_size = features[0].shape[1]
    padded_features = torch.zeros(len(features), max_len, hidden_size)
    attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)

    for i, f in enumerate(features):
        padded_features[i, :f.shape[0], :] = f
        attention_mask[i, :f.shape[0]] = 1

    labels = torch.stack(labels)
    return list(pids), padded_features, attention_mask, labels

def divide_sequence_dataset(ont: str, datapath: str):

    logger.info("Start loading data.")

    uniprot2string = get_uniprot2string(datapath)
    pid2index = get_ppi_pid2index(datapath)

    logger.info("Loading annotation data......")
    train_pid_list, train_label_list = get_pid_and_label_list(f'{datapath}/{ont}/{ont}_train_go.txt')
    valid_pid_list, valid_label_list = get_pid_and_label_list(f'{datapath}/{ont}/{ont}_valid_go.txt')
    test_pid_list, test_label_list = get_pid_and_label_list(f'{datapath}/{ont}/{ont}_test_go.txt')
    logger.info(F"Number of train pid: {len(train_pid_list)}, valid pid: {len(valid_pid_list)}, test pid: {len(test_pid_list)}.")

    logger.info('Get label matrix.')
    _, go_list = get_go_ic(os.path.join(datapath, ont, f'{ont}_go_ic.txt'))
    go_mlb = get_mlb(os.path.join(datapath, ont, f'{ont}_go.mlb'), go_list)

    train_true_label = go_mlb.transform(train_label_list).astype(np.float32)

    valid_true_label = go_mlb.transform(valid_label_list).astype(np.float32)

    test_true_label = go_mlb.transform(test_label_list).astype(np.float32)

    logger.info('load true labels done')


    train_pids_uniprot = get_pid_list(f'{datapath}/{ont}/{ont}_train.fasta')
    valid_pids_uniprot = get_pid_list(f'{datapath}/{ont}/{ont}_valid.fasta')
    test_pids_uniprot = get_pid_list(f'{datapath}/{ont}/{ont}_test.fasta')


    feature_dir = f"{datapath}/seq_feature"


    train_dataset = SeqDataset(train_pids_uniprot,feature_dir, train_true_label)
    valid_dataset = SeqDataset(valid_pids_uniprot,feature_dir, valid_true_label)
    test_dataset = SeqDataset(test_pids_uniprot,feature_dir, test_true_label)
    logger.info(f'load {ont} sequence datasets done')

    return train_dataset, valid_dataset, test_dataset