# -*- coding: utf-8 -*-
import os
import sys
from utils.func import *
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import networkx as nx
from model import *
import random

if __name__ == '__main__': 

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
        # print(arg['numUser'])
        classification_dataset = classificationDataset(arg['numUser'], dataSource, arg)

        # print(f"classification_dataset[0]: {classification_dataset[0]}")
        # print(f"classification_dataset[1]: {classification_dataset[1]}")
        classification_dataloader = DataLoader(classification_dataset, batch_size=arg['classification_batch'],
                                               shuffle=True, pin_memory=True,
                                               num_workers=0)
        # print('Data loaded')
        # print('init model')

        # print(arg['temporalGraph'])
        # print(arg['spatialGraph'])
        # print(arg['poi2geohash'+'_'+str(eachGeoHashPrecision)])
        # print(arg['geohash2poi'+'_'+str(eachGeoHashPrecision)])
        # print(arg['geohash2Index'+'_'+str(eachGeoHashPrecision)])
        # print(arg['beamSearchHashDict'])
        # print(arg['numUser'])