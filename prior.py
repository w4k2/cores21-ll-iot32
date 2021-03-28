import numpy as np
import matplotlib.pyplot as plt
import arff

names = ["33-1-2-43-1-3", "CTU-IoT-Malware-Capture-1-1_0", "CTU-IoT-Malware-Capture-33-1-p_2", "CTU-IoT-Malware-Capture-43-1-p_0", "CTU-IoT-Malware-Capture-43-1-p_3"]

for name in names:
    dataset = arff.load(open("arff/" + name + ".arff"))
    data = np.array(dataset['data'])
    y = data[:,-1].astype(int)

    chunk_samples = 250
    n_chunks = int(y.shape[0]/chunk_samples)
    print(n_chunks)

    start = 0
    stop = 0
    prior = np.zeros((2, n_chunks))
    for i in range(n_chunks):
        stop = start+chunk_samples
        # print("%i -- %i" % (start, stop))
        chunk_y = y[start:stop]
        start += chunk_samples
        classes, counts = np.unique(chunk_y, return_counts=True)

        if counts.shape[0] == 2:
            prior[0, i] = counts[0]/chunk_samples
            prior[1, i] = counts[1]/chunk_samples
        # if counts.shape[0] == 1 and 0 in classes:
        #     prior[0, i] = 1.0
        #     prior[1, i] = 0.0
        #     # chuj = i
        # if counts.shape[0] == 1 and 1 in classes:
        #     prior[0, i] = 0.0
        #     prior[1, i] = 1.0
    fig = plt.figure(figsize=(15, 5))
    plt.plot(prior[0], c="black", label="Benign")
    plt.plot(prior[1], c="red", label="Malicious")
    plt.xlim(0, n_chunks)
    plt.ylim(-0.006, 1.006)
    plt.title("%s\nPrior probabilities" % name, fontsize=16)
    plt.legend(ncol=2, loc=1, bbox_to_anchor=(1.0, 1.1), frameon=False, fontsize=12)
    plt.grid(ls="--", color=(0.85, 0.85, 0.85))
    plt.xlabel("Data chunk", fontsize=12)
    plt.ylabel("Prior", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/prior/%s.png" % (name), dpi=200)
    # plt.savefig("figures/%s_%i.eps" % (filename[:-4], p))
    plt.close()
