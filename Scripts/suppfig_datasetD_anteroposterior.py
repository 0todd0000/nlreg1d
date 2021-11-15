
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = True



# Dataset D:  COP (Pataky et al., 2014)
fnameCSV  = os.path.join( dirDATA, 'Pataky2014-anteroposterior.csv')
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
ylimt     = (-12, 12)
ylim      = [ (-1, 30), (-1, 30), (-20, 40), ylimt, (-2, 300), ylimt, ylimt ]
alpha_x   = [20, 30, 80, 80]
leg_loc   = [(0.32, 0.90), (0.32, 0.90), (0.99, 0.99)]
AX        = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_x, paired=True, dvlabel='Anteriorposterior COP  (cm)', group_labels=['Normal', 'Fast'], xlabel=xlabel, leg_loc=leg_loc)
tx0 = AX[0].text(0.5, 0.8, 'Anterior')
tx1 = AX[0].text(0.5, 0.1, 'Posterior')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=AX[0].transAxes, ha='center')
[ax.axhline(0, color='k', ls=':')  for ax in AX[:2]]
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'supp_datasetD_anteroposterior.pdf')  )


