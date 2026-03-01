
from collections import defaultdict
import os
from logzero import logger
import joblib
import subprocess
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from Bio import SeqIO
import pandas as pd

TAXIDS = ['3702','4577','6239','7227','7955','9031','9598','9606','10090','10116','44689','71421','284812','456442']

def get_mlb(mlb_path, classes = None, **kwargs) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit([classes])
    mlb.classes_ = np.array(classes, dtype=mlb.classes_.dtype)
    joblib.dump(mlb, mlb_path)
    return mlb

def get_pid_list(file_path):
    """
    file_path: fasta file path.
    """
    pid_list = list()
    for record in SeqIO.parse(file_path, 'fasta'):
        pid_list.append(record.id)
    return pid_list
    

def get_pid_and_label_list(file_path):
    pid_list, label_list = list(), list()
    tmp_dict = defaultdict(set)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            tmp_dict[line[0]].add(line[1])
    for pid in tmp_dict:
        pid_list.append(pid)
        label_list.append(list(tmp_dict[pid]))
    return pid_list, label_list

def get_label_dict(file_path):
    label_dict = dict()
    tmp_dict = defaultdict(set)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            tmp_dict[line[0]].add(line[1])
    for pid in tmp_dict:
        label_dict[pid] = list(tmp_dict[pid])
    return label_dict

def get_ppi_pid2index(datapath:str):
    pid2index = dict()
    with open(f"{datapath}/ppi_pid2index.txt") as f:
        for line in f:
            line = line.strip().split(' ')
            pid2index[line[0]] = int(line[1])   
    return pid2index

def readdata_ppi(filepath:str):
    proteins=set()
    with open(filepath, "r") as f:
        line0 = f.readline()
        lines = f.readlines()
        for line in lines:
            items = line.split(" ")
            proteins.add(items[0])
            proteins.add(items[1])
    return proteins

def readdata_mapid(filepath:str):
    data = pd.read_excel(filepath, sheet_name='Sheet0', usecols=["Entry","STRING"])
    string2uniprot={}
    for i in range(len(data['Entry'])):
        if pd.isnull(data['STRING'][i]) == False:
            ids = data['STRING'][i].strip(";").split(";")
            for id in ids:
                string2uniprot[id] = data['Entry'][i]
    return string2uniprot

def get_uniprot2string(datapath:str):
    if not os.path.exists(f"{datapath}/uniprot2string.txt"):
        uniprot2string={}
        spieces = TAXIDS

        for tax in spieces:
            string_ids = readdata_ppi(f'{datapath}/PPIdata/{tax}.protein.links.v12.0.txt')  ###########
            string2uniprot = readdata_mapid(f'{datapath}/IDmap-uniprot-string/uniprotkb_{tax}.xlsx')
            stringkeys = string2uniprot.keys()

            loseid_num=0
            
            for id in string_ids:
                if id in stringkeys:
                    uniprot2string[string2uniprot[id]] = id
                else:
                    loseid_num+=1
            logger.info(f'Here are {loseid_num} stringids lost')
        with open(f"{datapath}/uniprot2string.txt", "w") as f:
            for item in uniprot2string:
                f.write(f'{item} {uniprot2string[item]}\n')
        return uniprot2string
    uniprot2string = dict()
    with open(f"{datapath}/uniprot2string.txt") as f:
        for line in f:
            it = line.strip().split(' ')
            uniprot2string[it[0]] = it[1]
    return uniprot2string


def get_pid_ont(ont:str, datapath:str):
    pids = []
    for c in ['train', 'valid', 'test']:
        with open(f'{datapath}/{ont}/{ont}_{c}_pids.txt','r') as f:
            for line in f:
                pids.append(line.strip())
    return pids

def get_go_ic(go_ic_file_path):
    go_ic = dict()
    go_list = list()
    with open(go_ic_file_path) as f:
        for line in  f:
            line = line.strip().split('\t')
            go_ic[line[0]] = float(line[1])
            go_list.append(line[0])
    return go_ic, go_list


def get_network_index(uniprot2string, pid2index, pid_list, label_list, diamond_result=None, category='train'):
    index_list = list()
    tmp_pid_list = list()
    tmp_label_list = list()
    i = 0
    for pid, label in zip(pid_list, label_list):
        if pid2index.get(pid) != None:
            index_list.append(pid2index[pid])
            tmp_pid_list.append(pid)
            tmp_label_list.append(label)
        elif uniprot2string.get(pid) != None and pid2index.get(uniprot2string[pid]) != None:
            index_list.append(pid2index[uniprot2string[pid]])
            tmp_pid_list.append(pid)
            tmp_label_list.append(label)
        elif category == "eval" and diamond_result.get(pid) != None:
            index_list.append( pid2index[ max( diamond_result[pid].items(), key=lambda x: x[1] )[0] ] )
            tmp_pid_list.append(pid)
            tmp_label_list.append(label)
            i += 1
    if category == 'eval':
        logger.info(F"There are {i} proteins that don't have network index.")
    return np.asarray(index_list), tmp_pid_list, tmp_label_list


def get_pred_matrix(pred_file_path, pid2index, go2index) -> np.ndarray:
    pred_dict = defaultdict(list)
    with open(pred_file_path) as f:
        for line in f:
            line = line.strip().split('\t')
            pred_dict[line[0]].append((line[1], float(line[2])))
    pred_matrix = np.zeros((len(pid2index), len(go2index)), dtype=np.float32)
    for pid in pred_dict:
        for go_id, score in pred_dict[pid]:
            pred_matrix[pid2index[pid], go2index[go_id]] = score
    return pred_matrix


def diamond_homo(diamond_db, query_fasta_path, diamond_output_path):
    if not os.path.exists(diamond_output_path):
        logger.info("Run diamond.")
        cmd = ['diamond', 'blastp', '-d', diamond_db, '--very-sensitive', '-t', '/tmp', '-q', query_fasta_path, 
                '--outfmt', '6', 'qseqid', 'sseqid', 'evalue', 'bitscore', '-o', diamond_output_path]
        proc = subprocess.run(cmd)
    else:
        logger.info(F'Using exists diamond output file {diamond_output_path}')
    return parse_diamond_homo_result(diamond_output_path)


def parse_diamond_homo_result(diamond_output_path):
    diamond_sim = dict()
    with open(diamond_output_path) as f:
        for line in f:
            it = line.strip().split()
            evalue = float(it[2])
            if it[0] != it[1]:
                if it[0] not in diamond_sim:
                    diamond_sim[it[0]] = dict()
                diamond_sim[it[0]][it[1]] = float(it[3])
    return diamond_sim


def make_diamond_db(fasta, db_name):
    if not os.path.exists(db_name):
        logger.info(F"Making {db_name} database......")
        cmd = ['diamond', 'makedb', '--in', fasta, '-d', db_name]
        subprocess.run(cmd)
        logger.info("Done.")
    else:
        logger.info(F'diamond db {db_name} already exists.')

