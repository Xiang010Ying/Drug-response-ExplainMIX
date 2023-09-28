from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np


def draw():
    yLabel = ['ExplainMIX', 'ExplainNE', 'GnnExplainer']
    xLabel = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10']

    data = [[4 / 5, 4 / 5, 4 / 5, 4 / 5, 4 / 5, 5 / 5, 3 / 5, 2 / 5, 5 / 5, 4 / 5],
            [2 / 5, 3 / 5, 3 / 5, 1 / 5, 4 / 5, 4 / 5, 1 / 5, 4 / 5, 4 / 5, 3 / 5],
            [4 / 5, 3 / 5, 2 / 5, 0 / 5, 3 / 5, 4 / 5, 2 / 5, 2 / 5, 5 / 5, 2 / 5]]


    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    colors = ['#6AD4BE','#F5F2C5','#BC9BE0']
    mcolors.ListedColormap(colors)

    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.6, as_cmap=mcolors.LinearSegmentedColormap.from_list('cmap',colors,1000))

    im = ax.imshow(data, cmap=mcolors.LinearSegmentedColormap.from_list('cmap',colors,10))

    plt.colorbar(im)
    plt.yticks(rotation=30)

    plt.title("Comparing the Quality of Three Explanatory Methods",pad=18)

    data = np.array(data)
    for i in range(len(yLabel)):
        for j in range(len(xLabel)):
            ax.text(j, i, data[i, j],
                    ha="center", va="center", color="black")
    plt.show()


d = draw()
