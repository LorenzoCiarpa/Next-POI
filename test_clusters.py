# -*- coding: utf-8 -*-
import os
import sys
from utils.func_clusters import *
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import networkx as nx
from model_clusters import *
import random
import numpy as np

#reproducibility

# Imposta la seed per il generatore casuale di PyTorch
seed = 42
torch.manual_seed(seed)

# Imposta la seed per la generazione casuale CUDA (se usi GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Imposta la seed per tutti i dispositivi (se si utilizzano più GPU)

# Imposta la seed per il generatore casuale di Python
random.seed(seed)

# Imposta la seed per il generatore casuale di NumPy
np.random.seed(seed)

# Imposta alcuni parametri aggiuntivi per garantire la riproducibilità
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


with open('clusters.pkl', 'rb') as f:
    clusters = pickle.load(f)



if __name__ == '__main__': 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for dataSource in ['gowalla']:
        arg = {}

        start_time = time.time()

        arg['epoch'] = 20
        arg['beamSize'] = 100
        arg['embedding_dim'] = 1024
        arg['userEmbed_dim'] = 1024
        arg['hidden_dim']= 1024
        arg['classification_learning_rate'] = 0.0001
        arg['classification_batch'] = 32
        arg['dropout'] = 0.9
        arg['dataFolder'] = 'processedFiles'

        print()
        print(dataSource)
        print()
        print(arg)

        # ==================================Spatial Temporal graphs================================================
        arg['temporalGraph'] =  nx.read_edgelist('data/' + arg['dataFolder'] + '/' + dataSource + '_temporal.edgelist', nodetype=int,
                             create_using=nx.Graph())

        arg['spatialGraph'] = nx.read_edgelist(
            'data/' + arg['dataFolder'] + '/' + dataSource + '_spatial.edgelist', nodetype=int,
            create_using=nx.Graph())
        # ==================================Spatial Temporal graphs================================================

        userFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_userCount.pickle'

        with open(userFileName,
                  'rb') as handle:
            arg['numUser'] = pickle.load(handle)

        print('Data loading')

        # ==================================geohash related data================================================
        for eachGeoHashPrecision in [6,5,4,3,2]:
            poi2geohashFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poi2geohash' + '_' + str(
                eachGeoHashPrecision)
            geohash2poiFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2poi' + '_' + str(
                eachGeoHashPrecision)
            geohash2IndexFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2Index' + '_' + str(
                eachGeoHashPrecision)

            with open(poi2geohashFileName + '.pickle', 'rb') as handle:
                arg['poi2geohash'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2poiFileName + '.pickle', 'rb') as handle:
                arg['geohash2poi'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2IndexFileName + '.pickle', 'rb') as handle:
                arg['geohash2Index'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)

            arg['index2geoHash'+'_'+str(eachGeoHashPrecision)] = {v: k for k, v in arg['geohash2Index'+'_'+str(eachGeoHashPrecision)].items()}

        beamSearchHashDictFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_beamSearchHashDict'
        with open(beamSearchHashDictFileName + '.pickle', 'rb') as handle:
            arg['beamSearchHashDict'] = pickle.load(handle)
        # ==================================geohash related data================================================

        classification_dataset = classificationDataset(arg['numUser'], dataSource, arg)

        classification_dataloader = DataLoader(classification_dataset, batch_size=arg['classification_batch'],
                                               shuffle=True, pin_memory=True,
                                               num_workers=0)
        print('Data loaded')
        print('init model')

        classification = hmt_grn(arg).float().cuda()

        classification_optim = Adam(classification.parameters(), lr=arg['classification_learning_rate'])

        print('init model done')

        checkpoint = torch.load('checkpoint/GRN_cluster_20.pth')

        # Carica i pesi nel modello
        classification.load_state_dict(checkpoint)
        model = classification

        arg['novelEval'] = True
        evaluate(model, dataSource, arg)

        sys.stdout.flush()
