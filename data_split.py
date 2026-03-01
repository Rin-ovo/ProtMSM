import argparse
from collections import defaultdict
import requests as r
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import torch
import requests
from Bio import SeqIO
from io import StringIO
from Bio.SeqRecord import SeqRecord
from logzero import logger
from scipy.sparse import *
from scipy import *
import numpy as np
import logging
import json
from ontology import GeneOntology
from utils import *


TAXIDS = ['3702','4577','6239','7227','7955','9031','9598','9606','10090','10116','44689','71421','284812','456442']
func_dict = {
    'C': 'cc',
    'P': 'bp',
    'F': 'mf'
}
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP', 'IBA', 'IBD', 'IKR', 'IRD'])


FASTA_CACHE_DIR = "fasta_cache_claude"


def download_alphafold_structure(
        uniprot_id: str,
        out_dir: str,
        version: int = 4
):
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    uniprot_id = uniprot_id.upper()

    query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.pdb"
    structure_filename = os.path.join(out_dir, f"AF-{uniprot_id}-F1-model_v{version}.pdb")
    if os.path.exists(structure_filename):
        return structure_filename
    try:
        structure_filename = wget.download(query_url, out=out_dir)
    except:
        return None

    return structure_filename


def get_goa_spiece(datapath :str):
    all_data = {}
    spieces = TAXIDS

    for id in spieces:
        all_data[id ] =[]
    f = open(f"{datapath}/GO/goa_uniprot_all.gaf" ,"r")
    line = f.readline()
    while line.startswith('!'):
        line = f.readline()

    while line != '':
        data = line.strip().split('\t')
        taxon = data[12].strip().split(':')[1]
        eid = data[6]
        if taxon in spieces and eid in EXP_CODES:
            all_data[taxon].append \
                ({'PID' :f'{data[1]}', 'GO' :f'{data[4]}', 'EID' :f'{eid}', 'taxon' :f'{taxon}', 'aspect' :f'{data[8]}', 'time' :f'{data[13]}'})
            # print(f'PID:{data[1]}, GO:{data[4]}, EID:{eid}, taxon:{taxon}, aspect:{data[8]}, time:{data[13]}')
        line = f.readline()
    f.close()

    # output
    for id in all_data:
        with open(f'{datapath}/GO/goa_{id}.json' ,'w') as f:
            for item in all_data[id]:
                jsonitem = json.dumps(item)
                f.write(jsonitem)
                f.write('\n')
    return

def get_dataset(datapath :str):
    filename = f'{datapath}/GO/goa'
    go_file = f'{datapath}/GO/go.obo'

    spieces = TAXIDS


    onts = ['bp', 'mf', 'cc']
    cate = ['train', 'valid', 'test']
    data_set = dict()
    for o in onts:
        data_set[o] = dict()
        if not os.path.exists(f'{datapath}/{o}'):
            os.mkdir(f'{datapath}/{o}')
        for c in cate:
            data_set[o][c] = defaultdict(set)

    data_size_orgs = dict()
    for i in list(spieces):
        data_size_orgs[i] = dict()
        for o in onts:
            data_size_orgs[i][o] = dict()
            for c in cate:
                data_size_orgs[i][o][c] = set()

    uniprot2string = get_uniprot2string(datapath)
    prot_annots = defaultdict(dict)
    # get goa annotations
    # one example line of the file：
    # {"PID": "P54144", "GO": "GO:0043621", "EID": "IDA", "taxon": "3702", "aspect": "F", "time": "20170329"}

    for target in spieces:
        with open(f'{filename}_{target}.json', 'r') as f:
            for line in tqdm(f.readlines(), desc=f'Reading {target} annotations......'):
                line = json.loads(line)
                pid = line['PID']
                annot = line['GO']
                code = line['EID']
                ont = func_dict[line['aspect']]
                org_id = line['taxon']
                date = line['time']
                if code not in EXP_CODES:
                    continue
                if uniprot2string.get(pid) == None:
                    continue
                if date >= '20230801':
                    continue
                if prot_annots.get(pid) == None:
                    prot_annots[pid]['annotation'] = set()
                    prot_annots[pid]['annotation'].add( (annot, ont) )
                    prot_annots[pid]['org_id'] = org_id
                    prot_annots[pid]['min_date'] = date
                else:
                    prot_annots[pid]['annotation'].add( (annot, ont) )
                    prot_annots[pid]['min_date'] = min(prot_annots[pid]['min_date'], date)
        f.close()
    go = GeneOntology(go_file)
    labels_list = list()
    for pid in prot_annots:
        annot_set = set()
        for annot, _ in prot_annots[pid]['annotation']:
            annot_set |= go.get_ancestors(annot)
        annot_set -= ROOT_GO_TERMS
        labels_list.append(list(annot_set))
        annots = set()
        for annot in annot_set:
            annots.add( (annot, go.get_namespace(annot)) )
        prot_annots[pid]['annotation'] = annots # update

    go.calculate_ic(labels_list)
    logger.info('size of target protein: ' + str(len(prot_annots)))


    for pid, item in tqdm(prot_annots.items(), desc='split data set.......', total=len(prot_annots)):
        min_date = item['min_date']
        org_id = item['org_id']
        annots = item['annotation']
        if min_date >= '20220801' and min_date < '20230801':
            for annot, o in annots:
                data_size_orgs[org_id][o]['test'].add(pid)
                data_set[o]['test'][pid].add(annot)
        elif min_date >= '20210101' and min_date < '20220801':
            for annot, o in annots:
                data_size_orgs[org_id][o]['valid'].add(pid)
                data_set[o]['valid'][pid].add(annot)
        elif min_date < '20210101':
            for annot, o in annots:
                data_size_orgs[org_id][o]['train'].add(pid)
                data_set[o]['train'][pid].add(annot)

    if not os.path.exists(FASTA_CACHE_DIR):
        os.makedirs(FASTA_CACHE_DIR)


    seqs = {}
    for tax in list(spieces):
        if os.path.exists(f'{datapath}/sequence/uniprotkb_{tax}.fasta'):
            for record in SeqIO.parse(f'{datapath}/sequence/uniprotkb_{tax}.fasta', 'fasta'):
                pid = str(record.id).split('|')[1]
                seqs[pid] = record.seq

    logger.info('preprocess seqs.')

    for o in onts:
        for c in cate:
            for pid in tqdm(data_set[o][c], desc= 'download Sequences...'):
                fasta_file = os.path.join(FASTA_CACHE_DIR, f"{pid}.fasta")
                if pid in seqs:
                    continue

                if os.path.exists(fasta_file):
                    try:
                        with open(fasta_file, 'r') as f:
                            record = list(SeqIO.parse(f, 'fasta'))[0]
                            seqs[pid] = record.seq
                    except Exception as e:
                        logger.error(f"Error loading {pid} from cache in load fasta: {str(e)}")

                if pid not in seqs:
                    success = False

                    try:
                        baseUrl ="http://www.uniprot.org/uniprot/"
                        currentUrl =baseUrl +pid +".fasta"
                        response = r.post(currentUrl)
                        cData =''.join(response.text)


                        with open(fasta_file, 'w') as f:
                            f.write(cData)

                        Seq =StringIO(cData)
                        result = list(SeqIO.parse(Seq ,'fasta'))
                        seqs[pid ] =result[0].seq
                        success = True

                    except Exception as e:
                        logger.error(f"Error downloading {pid}: {str(e)} try another")

                    if not success:
                        try:
                            baseUrl = "https://rest.uniprot.org/uniprotkb/"
                            currentUrl = baseUrl + pid + ".fasta"
                            response = r.get(currentUrl)
                            cData = ''.join(response.text)
                            with open(fasta_file, 'w') as f:
                                f.write(cData)

                            Seq = StringIO(cData)
                            result = list(SeqIO.parse(Seq, 'fasta'))
                            seqs[pid] = result[0].seq
                            print("Download from NewUrl")
                            success = True
                        except Exception as e:
                            logger.error(f"Error downloading {pid}: {str(e)} for NewUrl")
    for o in onts:
        for c in cate:
            dataset_seqs = []
            dataset_pids = []
            for pid in tqdm(data_set[o][c], desc='split Sequences to dataset...'):
                dataset_seqs.append(SeqRecord(id=pid, seq=seqs[pid], description=''))
                dataset_pids.append(pid)
            ct = SeqIO.write(dataset_seqs, f'{datapath}/{o}/{o}_{c}.fasta', 'fasta')
            np.savetxt(f'{datapath}/{o}/{o}_{c}_pids.txt', dataset_pids, fmt='%s')



    logger.info('get all_seq.fasta')
    network_seqs = dict()
    index = 0
    ppi_pid2index = dict()
    for o in onts:
        for c in cate:
            for pid in data_set[o][c]:
                if pid not in network_seqs:
                    network_seqs[pid] = SeqRecord(id=uniprot2string[pid], seq=seqs[pid], description='')
                    ppi_pid2index[uniprot2string[pid]] = index
                    index += 1
    all_seqs_ct = SeqIO.write(list(network_seqs.values()), f'{datapath}/all_seq.fasta', 'fasta')

    with open(f'{datapath}/ppi_pid2index.txt', 'w') as f:
        for pid in ppi_pid2index:
            f.write(f'{pid} {ppi_pid2index[pid]}\n')
    print(f'len ppi_pid2index = {len(ppi_pid2index)}')
    assert len(ppi_pid2index) == all_seqs_ct


    logger.info('save goa data to _go.txt')
    for o in onts:
        for c in cate:
            with open(f'{datapath}/{o}/{o}_{c}_go.txt', 'w') as f:
                for pid in data_set[o][c]:
                    for go_id in data_set[o][c][pid]:
                        f.write(f"{pid}\t{go_id}\t{o}\t{prot_annots[pid]['org_id']}\n")
            f.close()



    logger.info('get go ic.')
    for o in onts:
        ont_go_set = set()
        for c in cate:
            for pid in data_set[o][c]:
                ont_go_set |= data_set[o][c][pid]
        sorted_go_list = list()
        for go_id in go.term_top_sort(o):
            if go_id in ont_go_set:
                sorted_go_list.append(go_id)
        assert len(ont_go_set) == len(sorted_go_list)
        with open(f'{datapath}/{o}/{o}_go_ic.txt', 'w') as f:
            for go_id in sorted_go_list:
                f.write(f'{go_id}\t{go.get_ic(go_id)}\n')
        f.close()

    with open(f'{datapath}/data_set_size.txt', 'w') as f:
        for o in onts:
            f.write(o + '\n')
            for c in cate:
                f.write('\t' + c + ' set: ' + str(len(data_set[o][c])) + '\n')
        f.write('\n')
        for i in list(spieces):
            f.write(str(i) + '\n')
            for o in onts:
                f.write('\t' + str(o) + '\n')
                for c in cate:
                    f.write('\t\t' + str(c) + ': ' + str(len(data_size_orgs[i][o][c])) + '\n')
    return

datapath="ProtMSM/data"
get_goa_spiece(datapath)
get_dataset(datapath)