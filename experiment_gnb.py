import numpy as np
from strlearn.evaluators import TestThenTrain
from strlearn.streams import ARFFParser
from strlearn.metrics import balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity
from strlearn.ensembles import LearnppCDS, LearnppNIE, OOB, UOB, WAE, OUSE, OnlineBagging, SEA, KMC
from sklearn.naive_bayes import GaussianNB
import multiprocessing

names = [
            # "CTU-IoT-Malware-Capture-1-1_0",
            "CTU-IoT-Malware-Capture-33-1-p_2",
            "CTU-IoT-Malware-Capture-43-1-p_0",
            # "CTU-IoT-Malware-Capture-43-1-p_3",
            # "33-1-2-43-1-3"
]
n_chunks = [
            # 4000,
            850,
            3300,
            # 1450,
            # 2300
]
metrics = (balanced_accuracy_score, geometric_mean_score_1, f1_score, precision, recall, specificity)

base = GaussianNB()

def worker(n, name):
    # print(name)
    filepath = ("arff/" + name + ".arff")
    # print(filepath)
    # stream
    stream = ARFFParser(filepath, chunk_size=250, n_chunks=n_chunks[n])
    # evaluator
    eval = TestThenTrain(metrics=metrics, verbose=False)
    # classifiers
    clfs = [
            SEA(base_estimator=base),
            OnlineBagging(base_estimator=base, n_estimators=10),
            OOB(base_estimator=base, n_estimators=10),
            UOB(base_estimator=base, n_estimators=10),
            LearnppCDS(base_estimator=base, n_estimators=10),
            LearnppNIE(base_estimator=base, n_estimators=10),
            OUSE(base_estimator=base, n_estimators=10),
            KMC(base_estimator=base, n_estimators=10),
            WAE(base_estimator=base, n_estimators=10),
            ]

    print("Started:  %s" % name)
    eval.process(stream, clfs)
    scores = eval.scores
    np.save("scores/%s-gnb" % name, scores)
    print("Finished: %s" % name)


jobs = []
for n, name in enumerate(names):
    p = multiprocessing.Process(target=worker, args=(n, name))
    jobs.append(p)
    p.start()


# for n, name in enumerate(names):
#     print(name)
#     filepath = ("arff/" + name + ".arff")
#     print(filepath)
#     # stream
#     stream = ARFFParser(filepath, chunk_size=250, n_chunks=n_chunks[n])
#     # evaluator
#     eval = TestThenTrain(metrics=metrics, verbose=True)
#     # classifiers
#     clfs = [
#             SEA(base_estimator=base),
#             OnlineBagging(base_estimator=base, n_estimators=10),
#             OOB(base_estimator=base, n_estimators=10),
#             UOB(base_estimator=base, n_estimators=10),
#             LearnppCDS(base_estimator=base, n_estimators=10),
#             LearnppNIE(base_estimator=base, n_estimators=10),
#             OUSE(base_estimator=base, n_estimators=10),
#             KMC(base_estimator=base, n_estimators=10),
#             WAE(base_estimator=base, n_estimators=10),
#             ]
#
#     eval.process(stream, clfs)
#     scores = eval.scores
#     np.save("scores/%s-gnb" % name, scores)
