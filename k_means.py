# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from utils.func import *
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import networkx as nx
from model import *
import random
import numpy as np
import pickle


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

def kmeans(data, k, max_iters=100, tol=1e-4):
    """
    Esegue l'algoritmo k-means sui dati.
    
    Args:
    - data: lista di numpy array, ogni array è lungo 3360 elementi
    - k: numero di cluster
    - max_iters: numero massimo di iterazioni
    - tol: tolleranza per la convergenza
    
    Returns:
    - centroids: array di centroidi
    - labels: array di etichette per ogni punto
    """
    # Convertiamo la lista di array in un array 2D
    # data = np.array(data)
    
    # Numero di punti
    num_points, num_features = data.shape
    
    # Inizializziamo i centroidi scegliendo k punti casuali dai dati
    centroids = data[np.random.choice(num_points, k, replace=False)]

    print(f"num_points: {num_points}")
    print(f"centroids: {centroids.shape}")
    
    for n_iter in range(max_iters):
        # if n_iter % 10 == 0:
        print(f"n_iter: {n_iter}")
        # Assegna ogni punto al centroide più vicino
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Calcola i nuovi centroidi
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        
        # Verifica la convergenza
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return centroids, labels


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

        # ==================================clustering users ================================================
        print("clustering users")

        users = {}

        data_root = 'data/' + arg['dataFolder'] + '/gowalla_train.pkl'

        with open('data/' + arg['dataFolder'] + '/' + dataSource + '_usersData.pickle',
                  'rb') as handle:
            userData = pickle.load(handle)
        trainUser, testUser = userData

        with open(data_root, 'rb') as f:
            pois_seq, delta_t_seq, delta_d_seq = cPickle.load(f)

        print(len(pois_seq))
        print(pois_seq[:1])
        print(len(userData[1]))


        # for i, seq in enumerate(pois_seq):
        #     for j, poi in enumerate(seq):
        #         print(i, j, poi, trainUser[i][j])
        #         if trainUser[i][j] not in users:
        #             users[trainUser[i][j]] = np.zeros(3360) # num_poi
                
        #         users[trainUser[i][j]][poi] += 1 
            
        # print(len(users))

        # Esempio di dizionario con array NumPy

        # # Salvare il dizionario su file
        # with open('data.pkl', 'wb') as f:
        #     pickle.dump(users, f)
        #     print("saved pickle file")

        with open('users_poi.pkl', 'rb') as f:
            users_poi = pickle.load(f)

        # # Estrai i valori dal dizionario
        # values = list(users_poi.values())

        # # Combina i valori in un array 2D
        # array_2d = np.vstack(values)

        # centroids, labels = kmeans(array_2d, k=20, max_iters=20)
        # print(centroids.shape)
        # print(labels.shape)

        # clusters = {
        #     "centroids": centroids,
        #     "labels": labels
        # }

        # with open('clusters.pkl', 'wb') as f:
        #     pickle.dump(clusters, f)
        #     print("saved pickle file")

        with open('clusters.pkl', 'rb') as f:
            clusters = pickle.load(f)

        
        labels = clusters['labels']
        labels_count = {}
        for i in labels:
            # print(i)
            if i not in labels_count:
                labels_count[i] = 0
            labels_count[i] += 1
        print(labels_count)