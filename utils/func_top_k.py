import torch as t
import numpy as np
import pickle
from tqdm import tqdm
import _pickle as cPickle


with open('clusters.pkl', 'rb') as f:
    clusters = pickle.load(f)

def evaluate(model, dataSource, arg):

    if dataSource == 'gowalla':
        data_root = 'data/' + arg['dataFolder'] + '/gowalla_train.pkl'
        save_dir = 'data/' + arg['dataFolder'] + '/gowalla_test.pkl'

    elif dataSource == 'global_scale':
        data_root = 'data/' + arg['dataFolder'] + '/global_scale_train.pkl'
        save_dir = 'data/' + arg['dataFolder'] + '/global_scale_test.pkl'

    with open(save_dir,'rb') as f:
        test_pois_seq, test_delta_t_seq, test_delta_d_seq = cPickle.load(f)

    if arg['novelEval'] == False: #for printing purposes
        print()
        print('Test File loaded')
        print()

    with open(data_root, 'rb') as f:
        train_pois_seq, train_delta_t_seq, train_delta_d_seq = cPickle.load(f)

    with open('data/' + arg['dataFolder'] + '/' + dataSource + '_usersData.pickle',
              'rb') as handle:
        userData = pickle.load(handle)
    trainUser, testUser = userData

    model.train(False)

    acc_K = [1, 5, 10, 20]

    result = {}
    totalMAP = []
    for K in acc_K:
        result[K] = 0

    totalTestInstances = 0


    for index in tqdm(range(len(test_pois_seq))):

        userAP = []
        test_set_poi_seq = [i for i in test_pois_seq[index]]
        test_set_user_seq = [i for i in testUser[index]]

        totalCheckins = 0
        hits = {}  #
        for K in acc_K:  # initialize the dict for each K.
            hits[K] = 0

        poi_test = test_set_poi_seq[:-1]
        user_test = test_set_user_seq[:-1]
        target_test = test_set_poi_seq[1:]

        for i in range(len(poi_test)):

            poiSeq = [poi_test[i]]
            userSeq = [user_test[i]]

            target_item = target_test[i]

            if arg['novelEval']:
                if target_item in train_pois_seq[index]: # we ignore the evaluation of this sample and continue to next.
                    continue

            totalCheckins += 1

            mappedGeoHash2 = arg['geohash2Index_2'][arg['poi2geohash_2'][poiSeq[0]]]
            mappedGeoHash3 = arg['geohash2Index_3'][arg['poi2geohash_3'][poiSeq[0]]]
            mappedGeoHash4 = arg['geohash2Index_4'][arg['poi2geohash_4'][poiSeq[0]]]
            mappedGeoHash5 = arg['geohash2Index_5'][arg['poi2geohash_5'][poiSeq[0]]]
            mappedGeoHash6 = arg['geohash2Index_6'][arg['poi2geohash_6'][poiSeq[0]]]

            input = [poiSeq, userSeq, [[mappedGeoHash2]], [[mappedGeoHash3]], [[mappedGeoHash4]],
                     [[mappedGeoHash5]], [[mappedGeoHash6]], clusters]

            userID = index + 1
            assert userID == userSeq[0]  #  double check


            pred, nextgeohashPred_2_test, nextgeohashPred_3_test, nextgeohashPred_4_test, nextgeohashPred_5_test, nextgeohashPred_6_test = model(
                input, 'test', arg)

            pred = pred.view(-1)
            pred = pred[1:]
            target_item -= 1
            sortedPreds = t.topk(pred, len(pred))[1].tolist()

            temptHistoryPOIs = [i - 1 for i in list(set(train_pois_seq[index]))]
            # =====================Hierarchical Beam Search====================

            if sortedPreds[0] not in temptHistoryPOIs or arg['novelEval']:  # unvisited

                allSequenceDict = {}
                sequencesDict = {}

                for iterationIndex in [2, 3, 4, 5, 6]:
                    all_candidates = []
                    if iterationIndex == 2:
                        row = nextgeohashPred_2_test[0][0]
                        try:
                            topBeam = t.topk(row, arg['beamSize'])
                        except:
                            topBeam = t.topk(row, len(row))
                        topBeam_indices = topBeam.indices.tolist()
                        topBeam_Prob = topBeam.values.tolist()
                        wholeSeqList = topBeam_indices
                    else:

                        topBeam_indices = [i[0][-1:][0] for i in sequencesDict[iterationIndex - 1]]
                        topBeam_Prob = [i[1] for i in sequencesDict[iterationIndex - 1]]

                        wholeSeqList = [i[0] for i in sequencesDict[iterationIndex - 1]]

                    if 0 in topBeam_indices: #remove 0 padding
                        topBeam_indices.remove(0)

                    # get sub-nodes
                    for eachTopK, prob, pastSeqList in zip(topBeam_indices, topBeam_Prob, wholeSeqList):
                        beforeHash = arg['index2geoHash' + '_' + str(iterationIndex)][eachTopK]
                        if iterationIndex == 6:
                            mappedGeoHash_last = arg['index2geoHash_6'][eachTopK]
                            geoHashPOIs_last = arg['geohash2poi_6'][mappedGeoHash_last]
                            subNodes2Index = [i - 1 for i in geoHashPOIs_last]
                        else:
                            currentPrecisionRelation = str(iterationIndex) + '_' + str(iterationIndex + 1)
                            subNodes = arg['beamSearchHashDict'][currentPrecisionRelation][beforeHash]
                            # map sub-nodes to index
                            subNodes2Index = [arg['geohash2Index' + '_' + str(iterationIndex + 1)][i] for i in
                                              subNodes]
                        # get their probabilities from prediction
                        if iterationIndex == 2:
                            geohashPredChoice = nextgeohashPred_3_test
                        elif iterationIndex == 3:
                            geohashPredChoice = nextgeohashPred_4_test
                        elif iterationIndex == 4:
                            geohashPredChoice = nextgeohashPred_5_test
                        elif iterationIndex == 5:
                            geohashPredChoice = nextgeohashPred_6_test
                        elif iterationIndex == 6:
                            geohashPredChoice = pred

                        if type(pastSeqList) is int:
                            pastSeqList = [pastSeqList]

                        subNodes_Probs = geohashPredChoice.view(-1)[subNodes2Index].cpu().detach().numpy()

                        for eachSubNodeIndex in range(len(subNodes2Index)):
                            # eachSubNodeIndex
                            candidate = [pastSeqList + [subNodes2Index[eachSubNodeIndex]],
                                         prob + np.log(subNodes_Probs[eachSubNodeIndex])]
                            all_candidates.append(candidate)

                    ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)

                    output = ordered[:arg['beamSize']]

                    allSequenceDict[iterationIndex] = ordered
                    sequencesDict[iterationIndex] = output


                allCandidate_lastPOIs = [i[0][-1:][0] for i in ordered]

                remaining = set(range(pred.shape[0])) - set(allCandidate_lastPOIs)
                sortedPreds = list(allCandidate_lastPOIs) + list(remaining)

            # =====================Hierarchical Beam Search====================

            # for novel eval, to remain only novel recommendations, we remove the user's visited POI from the distribution or ranked list
            if arg['novelEval']:
                historyPOIs = list(set(train_pois_seq[index]))
                historyPOIs = [i - 1 for i in historyPOIs]
                sortedPreds = [i for i in sortedPreds if i not in historyPOIs]

            truthIndex = sortedPreds.index(target_item) + 1

            averagePrecision = 1 / truthIndex

            userAP.append(averagePrecision)

            sorted_indexs = {}
            for K in acc_K:
                sorted_indexs[K] = sortedPreds[:K]

            # Check if ground truth in top K for each acc@K
            for K in acc_K:
                if target_item in sorted_indexs[K]:
                    hits[K] += 1

        totalTestInstances += totalCheckins

        for K in acc_K:
            result[K] += hits[K]

        userMAP = userAP

        totalMAP = totalMAP + userMAP

    for K in acc_K:
        result[K] /= totalTestInstances

    print(str(result[1]) + ',' + str(result[5]) + ',' + str(result[10]) + ',' + str(result[20]) + ',' + str(
        np.average(totalMAP)), end='')
    print()

    if arg['novelEval'] == True:  # for printing neatly.
        print()
        print(dataSource + ' Finished!')
        print()

    model.train(True)



def get_top_k_users(users_clustered, cluster_id, id_user, top_k = 5):

    top_5_vectors = getMostSimilar(users_clustered, cluster_id, id_user, top_k)
    
    users_clustered_t = users_clustered[cluster_id][:top_k]
    # print(f"users_clustered_t 1: {users_clustered_t}")
    users_clustered_t = t.tensor(np.array(users_clustered_t))

    # print(f"top_5_vectors: {top_5_vectors}")
    # print(f"users_clustered_t: {users_clustered_t}")

    # print(f"users_clustered_t 2: {users_clustered_t}")
    users_clustered_t = users_clustered_t.mean(axis=0)
    # print(f"users_clustered_t 3: {users_clustered_t.shape}")

    return users_clustered_t

def getMostSimilar(users_clustered, cluster_id, id_user, top_k):
    users_clustered_t = users_clustered[cluster_id]
    users_clustered_t = t.tensor(np.array(users_clustered_t))
    user = users_clustered_t[0]

    # print(f"getMost users_clustered_t: {users_clustered_t}")
    # print(f"getMost user: {user}")

    # Normalizza i vettori per il calcolo della similarità del coseno
    v_norm = user / t.norm(user, dim=0, keepdim=True)
    m_norm = users_clustered_t / t.norm(users_clustered_t, dim=1, keepdim=True)

    if users_clustered_t.shape[0] < 5:
        return users_clustered_t
    
    # Calcola la similarità del coseno
    similarity = t.matmul(m_norm, v_norm.t()).squeeze()

    # Ottieni gli indici dei 5 vettori più simili
    _, indices = t.topk(similarity, 5, largest=True)

    # Filtra la matrice m sui 5 indici più simili
    top_5_vectors = users_clustered_t[indices]

    return top_5_vectors


if __name__ == '__main__':
    a = t.ones(3)
    b = t.ones(3) * 2
    c = t.ones(3) * 3
    d = t.ones(3) * 4
    # print(t.cat(a))
    e = t.tensor([[a, b], [c, d]])
    print(e)

