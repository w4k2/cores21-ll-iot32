import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from math import pi
from scipy.ndimage.filters import gaussian_filter1d
import os
from matplotlib import rcParams

# Set plot params
rcParams["font.family"] = "monospace"
# methods = ["SEA", "OB", "OOB", "UOB", "L++CDS", "L++NIE", "OUSE", "KMC", "WAE"]
# colors = ["black", "red", "red", "red", "green", "green", "blue", "blue", "blue",]
# ls = ["-", "-", "--", ":", "-", "--", "-", "--", ":"]
methods = ["OB", "OOB", "UOB", "L++CDS", "L++NIE", "WAE"]
colors = ["red", "red", "red", "green", "green", "blue"]
ls = ["-", "--", ":", "-", "--", "-"]
lw = [1, 1, 1, 1, 1, 1, 1, 1, 1]
metrics = ["Balanced_accuracy", "G-mean", "f1 score", "precision", "recall", "specificity"]
metrics2 = ["BAC", "$Gmean_s$", "$F_1$ $score$", "$precision$", "$recall$", "$specificity$"]


which = "gnb"
real = [
    "CTU-IoT-Malware-Capture-43-1-p_3",
    "CTU-IoT-Malware-Capture-1-1_0",
    "CTU-IoT-Malware-Capture-33-1-p_2",
    "33-1-2-43-1-32",
    "CTU-IoT-Malware-Capture-43-1-p_0",
]
n_chunks = [
            1450,
            4000,
            850,
            2300,
            2500,
]

# which = "ht"
# real = [
#     "CTU-IoT-Malware-Capture-43-1-p_3",
#     "CTU-IoT-Malware-Capture-33-1-p_2",
#     "CTU-IoT-Malware-Capture-1-1_0",
#     "CTU-IoT-Malware-Capture-43-1-p_0",
#     "33-1-2-43-1-3",
# ]
# n_chunks = [
#             1450,
#             850,
#             4000,
#             3300,
#             2300,
# ]


def plot_runs(metrics, selected_scores, methods, mean_scores, what
):
    # fig = plt.figure(figsize=(5, 3))
    fig = plt.figure(figsize=(8, 5.5))
    ax = plt.axes()
    for z, (value, label, mean) in enumerate(
        zip(selected_scores, methods, mean_scores)
    ):
        label = "\n{0:.3f}".format(mean)
        val = gaussian_filter1d(value, sigma=31, mode="nearest")
        # val = medfilt(value, kernel_size=41)

        # plt.plot(value, label=label, c=colors[z], ls=ls[z])

        plt.plot(val, label=label, c=colors[z], ls=ls[z], lw=lw[z])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    ax.legend(
        loc=8,
        bbox_to_anchor=(0.5, 0.97),
        fancybox=False,
        shadow=True,
        ncol=3,
        # fontsize=7,
        fontsize=13,
        frameon=False,
    )

    plt.grid(ls=":", c=(0.7, 0.7, 0.7))
    plt.xlim(0, n_chunks[j])
    axx = plt.gca()
    axx.spines["right"].set_visible(False)
    axx.spines["top"].set_visible(False)

    # plt.title(
    #     "%s %s\n%s" % (what, "GNB", metrics[i]),
    #     fontfamily="serif",
    #     y=1.04,
    #     fontsize=8,
    # )
    plt.ylim(0.0, 1.0)
    # plt.xticks(fontsize=9)
    # plt.yticks(fontsize=9)
    # plt.ylabel("%s" % metrics2[i], fontsize=10)
    # plt.xlabel("chunks", fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("%s" % metrics2[i], fontsize=13)
    plt.xlabel("chunks", fontsize=13)
    plt.tight_layout()
    if metrics[i] == "G-mean":
        plt.savefig("figures/runs/%s_%s_%s.eps" % (what, metrics[i], which), bbox_inches='tight', dpi=250)
    plt.close()

def plot_radars(
    methods, metrics, table, classifier_name, parameter_name, what
):
    """
    Strach.
    """
    columns = ["group"] + methods
    df = pd.DataFrame(columns=columns)
    for i in range(len(table)):
        df.loc[i] = table[i]
    df = pd.DataFrame()
    df["group"] = methods
    for i in range(len(metrics)):
        df[table[i][0]] = table[i][1:]
    groups = list(df)[1:]
    N = len(groups)

    print(df.to_latex(index=False))

    # nie ma nic wspolnego z plotem, zapisywanie do txt texa
    # print(df.to_latex(index=False), file=open("tables/%s.tex" % (filename), "w"))

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # No shitty border
    ax.spines["polar"].set_visible(False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, metrics2)

    # Adding plots
    for i in range(len(methods)):
        values = df.loc[i].drop("group").values.flatten().tolist()
        values += values[:1]
        values = [float(i) for i in values]
        ax.plot(
            angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
        )

    # Add legend
    plt.legend(
        loc="lower center",
        ncol=3,
        columnspacing=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.32),
        fontsize=6,
    )

    # Add a grid
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))

    # Add a title
    plt.title("%s" % (what), size=8, y=1.10)
    plt.tight_layout()

    # Draw labels
    a = np.linspace(0, 1, 6)
    plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.gcf().set_size_inches(4, 3.5)
    plt.gcf().canvas.draw()
    angles = np.rad2deg(angles)

    ax.set_rlabel_position((angles[0] + angles[1]) / 2)

    har = [(a >= 90) * (a <= 270) for a in angles]

    for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
        x, y = label.get_position()
        # print(label, angle)
        lab = ax.text(
            x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
        )
        lab.set_rotation(angle)

        if har[z]:
            lab.set_rotation(180 - angle)
        else:
            lab.set_rotation(-angle)
        lab.set_verticalalignment("center")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
        x, y = label.get_position()
        # print(label, angle)
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            fontsize=4,
            c=(0.7, 0.7, 0.7),
        )
        lab.set_rotation(-(angles[0] + angles[1]) / 2)

        lab.set_verticalalignment("bottom")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # print("TUTAJ:  ",parameter_name)
    plt.savefig("figures/radars/%s_%s.eps" % (what, which), bbox_inches='tight', dpi=250)
    plt.close()

j = 0
for i, (root, dirs, files) in enumerate(os.walk("scores/"+which+"/")):
    for filename in files:
        filepath = root + os.sep + filename
        if filepath.endswith(".npy"):
            print(filename)
            scores = np.load(filepath)
            scores = scores[[1,2,3,4,5,8]]
            # print(scores.shape)
            # exit()
            for i, metric in enumerate(metrics):
                # print("\n---\n--- %s\n---\n" % (metric))
                # METHOD, CHUNK, METRYKA
                selected_scores = scores[:, :, i]
                mean_scores = np.mean(selected_scores, axis=1)

                plot_runs(metrics, selected_scores, methods, mean_scores, real[j])

            # RADAR DIAGRAMS
            table = []
            header = ["Metric"] + methods
            for i, metric in enumerate(metrics):
                # METHOD, CHUNK, Metryka
                selected_scores = scores[:, :, i]
                mean_scores = np.mean(selected_scores, axis=1)
                table.append([metric] + ["%.3f" % score for score in mean_scores])

            # print(tabulate(table, headers=header, tablefmt="latex_booktabs"))
            plot_radars(methods, metrics, table, "", "", real[j])
            j += 1
