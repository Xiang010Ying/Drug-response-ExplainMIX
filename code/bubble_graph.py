import matplotlib.pyplot as plt

# 1940行 [527, 0.37579796, 0, 171]
jaccard = [0.500000, 0.666667, 1.000000+0.005, 0.500000+0.005, 0.833333, 0.833333, 0.500000, 0.666667, 0.500000+0.005, 0.500000, 0.500000, 0.666667, 1.000000, 0.500000, 0.666667]
corr = [0.094574, 0.083632, 0.106442, 0.029494, 0.149970, 0.108808, 0.075712, 0.078068, 0.048343, 0.015457, 0.048926, 0.040295, 0.106442, 0.029494, 0.065998]
p = [5, 4, 6, 1, 10, 6, 3, 3, 2, 1, 2, 2, 6, 1, 3]
p1 = [i*300 for i in p]
target = ['ExplainMIX','ExplainMIX','ExplainMIX','ExplainMIX','ExplainMIX','ExplainNE','ExplainNE','ExplainNE','ExplainNE','ExplainNE','GnnExplainer','GnnExplainer','GnnExplainer','GnnExplainer','GnnExplainer']

colors = ['#B783AF', '#B783AF', '#B783AF', '#B783AF', '#B783AF','#736B9D','#736B9D','#736B9D','#736B9D','#736B9D','#365083','#365083','#365083','#365083','#365083']  # 颜色数组
size = p1

plt.scatter(jaccard[0:5], corr[0:5], s=size[0:5], c='#FCDB72', alpha=0.8, edgecolors='#FCDB72',label='ExplainMIX')
plt.scatter(jaccard[5:10], corr[5:10], s=size[5:10], c='#B783AF', alpha=0.6, edgecolors='#B783AF',label='ExplainNE')
plt.scatter(jaccard[10:15], corr[10:15], s=size[10:15], c='#736B9D', alpha=0.6, edgecolors='#736B9D',label='GnnExplainer')

plt.ylim([0, 0.2])
plt.xlim([0.2, 1.2])
plt.xlabel('Jaccard Correlation',fontsize = 12)
plt.ylabel('Kendall Correlation',fontsize = 12)

# Label the scatter points
for i in range(len(jaccard)):
    if i//5 == 0:
        plt.annotate(target[i], (jaccard[i], corr[i]), (jaccard[i]-0.18, corr[i]-.025),color = "#11325D",weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
    if i//5 == 1:
        if i == 5:
            plt.annotate(target[5], (jaccard[5], corr[5]),(jaccard[5]+0.01*i, corr[5]+.005*i),color = "#11325D",weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
        if i == 6:
            plt.annotate(target[6], (jaccard[6], corr[6]), (jaccard[6] + 0.065, corr[6] + .03), color="#11325D",
                     weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
        if i == 7:
            plt.annotate(target[7], (jaccard[7], corr[7]), (jaccard[7] + 0.1, corr[7] - .01), color="#11325D",
                         weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
        if i == 8:
            plt.annotate(target[8], (jaccard[8], corr[8]), (jaccard[8] - 0.2, corr[8] + .005), color="#11325D",
                         weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
        if i == 9:
            plt.annotate(target[9], (jaccard[9], corr[9]), (jaccard[9] - 0.25, corr[9] + .005), color="#11325D",
                         weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
        #plt.annotate(target[i], (jaccard[i], corr[i]), (jaccard[i]+0.01*i, corr[i]+.005*i),color = "#11325D",weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))
    if i//5 == 2:
        plt.annotate(target[i], (jaccard[i], corr[i]), (jaccard[i]+0.0006, corr[i]-.002*i),color = "#11325D",weight='light', arrowprops=dict(facecolor='#11325D', arrowstyle='->'))

plt.legend(loc='upper left',markerscale=0.5,fontsize=15)
plt.show()

# 1533行 [197, 0.5146, 1, 85]
jaccard = [0.666667, 0.500000, 1, 0.333333, 1.000000+0.0005]
corr = [0.182973, 0.111059, 0.438243, 0.026281, 0.441516]
jaccard_ = [1, 1.000000+0.01]
corr_ = [0.438243, 0.441516]
size_ = [38*30,39*30]

p = [2, 1, 38, 1, 39]
p1 = [i*50 for i in p]
target = ['ExplainMIX','ExplainMIX','GnnExplainer','GnnExplainer','GnnExplainer']

colors = ['#B783AF', '#B783AF','#365083','#365083','#365083']  # 颜色数组
markes = ['^','o','^']
size = p1

plt.scatter(jaccard[0:2], corr[0:2], s=size[0:2], c='#FCDB72', alpha=0.8, edgecolors='#FCDB72',label='ExplainMIX')
plt.scatter(jaccard[3], corr[3], s=size[3], c='#736B9D', alpha=0.6, edgecolors='#736B9D',label='GnnExplainer(1)')
plt.scatter(jaccard_, corr_, s=size_, c='#736B9D', alpha=0.6, edgecolors='#736B9D',label='GnnExplainer(0)',marker='^')

plt.ylim([0, 0.5])
plt.xlim([0.2, 1.2])

plt.xlabel('Jaccard Correlation',fontsize = 12)
plt.ylabel('Kendall Correlation',fontsize = 12)


plt.legend(loc='upper left',markerscale=0.5,fontsize=15)

plt.show()
