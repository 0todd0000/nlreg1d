
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = True



# Dataset D:  COP (Pataky et al., 2014)
fnameCSV  = os.path.join( dirDATA, 'Pataky2014-mediolateral.csv')
a         = np.loadtxt( fnameCSV, delimiter=',')
g,y       = a[:,0], a[:,1:]  # group, dependent variable
J         = (g==0).sum()     # number of observations in first group
### register:
Q         = y.shape[1]
q         = np.linspace(0, 1, Q)
yr,wr     = nlreg1d.reg.fpca( y, ncomp=5, smooth=False, niter=niter )
### plot:
plt.close('all')
colors    = '0.0', (0.8,0.1,0.1)
xlabel    = 'Time  (% stance)'
ylimt     = (-7, 7)
ylim      = [ (-2, 3), (-2, 3), (-35, 48), ylimt, (-2, 60), ylimt, ylimt ]
alpha_x   = [20, 20, 80, 80]
AX        = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_x, paired=True, dvlabel='Mediolateral COP  (cm)', group_labels=['Normal', 'Fast'], xlabel=xlabel)
tx0 = AX[0].text(0.1, 0.8, 'Lateral')
tx1 = AX[0].text(0.1, 0.1, 'Medial')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=AX[0].transAxes)
[ax.axhline(0, color='k', ls=':')  for ax in AX[:2]]
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'datasetD.pdf')  )
