import numpy as np
from strlearn.evaluators import TestThenTrain
from strlearn.streams import ARFFParser
from strlearn.metrics import balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity
from strlearn.ensembles import LearnppCDS, LearnppNIE, OOB, UOB, WAE, OUSE, OnlineBagging, SEA, KMC
from sklearn.naive_bayes import GaussianNB
from skmultiflow.trees import HoeffdingTree
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

names = [
            "CTU-IoT-Malware-Capture-1-1_0",
            "CTU-IoT-Malware-Capture-33-1-p_2",
            "CTU-IoT-Malware-Capture-43-1-p_0",
            "CTU-IoT-Malware-Capture-43-1-p_3",
            "33-1-2-43-1-3"
]
n_chunks = [
            4000,
            850,
            3300,
            1450,
            2300
]

# metrics = (balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity)

for n, name in enumerate(names):
    filepath = ("arff/" + name + ".arff")
    stream = ARFFParser(filepath, chunk_size=250, n_chunks=n_chunks[n])
    clf = SEA(base_estimator=GaussianNB())
    clf2 = HoeffdingTree(split_criterion='hellinger')

    X, y = stream.get_chunk()
    clf.fit(X, y)
    clf2.fit(X,y)

    probas = []
    probas2 = []
    ys = []
    # limit = 150
    for chunk in tqdm(range(n_chunks[n]-1)):
        X, y = stream.get_chunk()
        proba = clf.predict_proba(X)[:,1]
        try:
            proba2 = clf2.predict_proba(X)[:,1]
        except:
            try:
                proba2 = np.concatenate(proba2)
            except:
                pass

        clf.partial_fit(X, y)
        clf2.partial_fit(X, y)
        probas.append(proba)
        probas2.append(proba2)
        ys.append(y)
        # if chunk == limit:
        #     break

    probas = np.array(probas)
    probas = probas.reshape(-1)
    probas2 = np.array(probas2)
    probas2 = probas2.reshape(-1)
    ys = np.array(ys)
    ys = ys.reshape(-1)
    ls = (np.linspace(0,1,probas.shape[0]))
    # print(ls.shape)
    # print(probas.shape)

    fig, ax = plt.subplots(1,2, figsize=(10*1.618, 5))
    ax[1].scatter(ls, probas2, c=ys, cmap="bwr", alpha=.1)
    ax[1].set_title("HT")
    ax[0].scatter(ls, probas, c=ys, cmap="bwr", alpha=.1)
    ax[0].set_title("GNB")
    plt.tight_layout()
    plt.savefig("figures/probas/%s" % name)
    plt.savefig("foo")
