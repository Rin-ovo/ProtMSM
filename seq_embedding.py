import argparse
from logzero import logger
import shutil
import torch
torch.cuda.empty_cache()
import esm
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import os
import re
from utils import get_uniprot2string, get_ppi_pid2index
"""
esm-extract esm2_t33_650M_UR50D examples/data/some_proteins.fasta \
  examples/data/some_proteins_emb_esm2 --repr_layers 0 32 33 --include

@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
"""

def chunk_seq(seq, max_len=1024, overlap=200):
    chunks = []
    start = 0
    while start < len(seq):
        end = min(start + max_len, len(seq))
        chunks.append(seq[start:end])
        if end == len(seq):
            break
        start = end - overlap
    return chunks


# Load ESM-2 model
def get_seq_feature(datapath:str, modelfile:str, max_len:int=1024, overlap:int=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(modelfile)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    if not os.path.exists(f'{datapath}/seq_feature'):
        os.mkdir(f'{datapath}/seq_feature')

    files_done = [file.split('.')[0] for file in os.listdir(f'{datapath}/seq_feature')]
    seqs_done = set(files_done)

    files = sorted(
        [file for file in os.listdir(f'{datapath}/split_seqs') if file.startswith('network_')],
        key=lambda x: int(re.search(r'network_(\d+)\.fasta', x).group(1)) if re.search(r'network_(\d+)\.fasta', x) else 0
    )

    for file in files:
        logger.info(f'{file} to be processing')
        seqs = []
        for record in SeqIO.parse(f'{datapath}/split_seqs/{file}', 'fasta'):
            if str(record.id) in seqs_done:
                continue
            seqs.append((str(record.id), str(record.seq)))
        if len(seqs) == 0:
            continue

        for pid, seq in tqdm(seqs, desc='esm2 embedding...'):
            if len(seq) > max_len:
                chunks = chunk_seq(seq, max_len, overlap)
                feats = []
                for c in chunks:
                    data = [(pid, c)]
                    batch_labels, batch_strs, batch_tokens = batch_converter(data)
                    batch_tokens = batch_tokens.to(device)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

                    token_representations = results["representations"][33].cpu()
                    tokens_len = (batch_tokens != alphabet.padding_idx).sum(1)[0].item()
                    feats.append(token_representations[0, 1:tokens_len-1].numpy())

                seq_feat = np.concatenate(feats, axis=0)
            else:
                data = [(pid, seq)]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                tokens_len = (batch_tokens != alphabet.padding_idx).sum(1)[0].item()
                seq_feat = results["representations"][33][0, 1:tokens_len-1].cpu().numpy()

            np.save(f'{datapath}/seq_feature/{pid}.npy', seq_feat)
            seqs_done.add(pid)


def split_network_seqs(datapath:str):
    string2uniprot = {}
    uniprot2string = get_uniprot2string(datapath)
    for uniprotid in uniprot2string:
        string2uniprot[uniprot2string[uniprotid]] = uniprotid

    seqs = dict()
    for record in SeqIO.parse(f'{datapath}/network.fasta', 'fasta'):
        seqs[string2uniprot[str(record.id)]] = str(record.seq)

    length = 250
    
    split_seqs = dict()
    for i in range(len(seqs)//length+1):
        split_seqs[str(i)] = []
    index = 0
    for pid in seqs:
        split_seqs[str(index//length)].append( SeqRecord(id=pid, seq=seqs[pid], description='') )
        index += 1

    if not os.path.exists(f'{datapath}/split_seqs'):
        os.mkdir(f'{datapath}/split_seqs')

    for i in split_seqs:
        ct = SeqIO.write(split_seqs[i], f'{datapath}/split_seqs/network_{i}.fasta', 'fasta')
    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument("-d", "--datapath", type=str, default='ProtMSM/data',
            help="path to save data")
    parser.add_argument("-m", "--modelfile", type=str, default='./esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt',
            help="The path of ESM-2 model .pt file")
    torch.cuda.empty_cache()
    args = parser.parse_args()
    logger.info(args)
    # get seq embedding
    #split_network_seqs(args.datapath)
    get_seq_feature(args.datapath, args.modelfile)
    #get_seq_embedding(args.datapath)


