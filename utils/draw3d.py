import matplotlib as mpl
import os
import numpy as np

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def drawpc(suffix, point_gt, point_pr):
    figg = plt.figure(figsize=(10, 10), dpi=200)
    figp = plt.figure(figsize=(10, 10), dpi=200)
    axg = Axes3D(figg)
    axp = Axes3D(figp)
    for i in range(len(point_gt)):
        gt = np.asarray(point_gt[i])
        pr = np.asarray(point_pr[i])
        print gt.shape
        axg.scatter(gt[:,0], gt[:,1], gt[:,2],\
                    color='C{}'.format(i), s = 1, alpha=1)
        axp.scatter(pr[:,0], pr[:,1], pr[:,2],\
                    color='C{}'.format(i), s = 1, alpha=1)
    figg.savefig('{}_gt.png'.format(suffix), bbox_inches='tight')
    figp.savefig('{}_pd.png'.format(suffix), bbox_inches='tight')
    plt.close()