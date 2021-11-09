
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = True



# Dataset C:  vastus forces (Besier et al., 2009):
fnameCSV  = os.path.join( dirDATA, 'Besier2009-vastus.csv')
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
ylim      = [ (-0.05, 1.3), (-0.05, 1.3), (-25, 39), (-6, 6), (-2, 30), (-6,6), (-6,6) ]
alpha_loc = [  [(70,3.5), (70,2.6)]  , [(70,17), (70,14.5)]  ,  [(70,3.7), (70,2.8)]   ,    [(70,3.8), (70,2.9)]  ]
leg_loc   = [(0.80, 0.92), (0.80, 0.92), (0.99, 0.99)]
AX        = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_loc, paired=False, dvlabel='Muscle force  (kN)', group_labels=['Control mean', 'PFP mean'], xlabel=xlabel, leg_loc=leg_loc)
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'datasetC.pdf')  )



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
ylim      = [ (-4, 6), (-4, 6), (-35, 48), ylimt, (-2, 60), ylimt, ylimt ]
alpha_loc = [  [(20,4.7), (20,3.8)]  , [(20,38), (20,33)]  ,  [(80,4.8), (80,3.7)]   ,    [(80,5.3), (80,4.1)]  ]
AX        = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_loc, paired=True, dvlabel='Mediolateral COP  (cm)', group_labels=['Normal', 'Fast'], xlabel=xlabel)
tx0 = AX[0].text(0.1, 0.8, 'Lateral')
tx1 = AX[0].text(0.1, 0.1, 'Medial')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=AX[0].transAxes)
[ax.axhline(0, color='k', ls=':')  for ax in AX[:2]]
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'datasetD.pdf')  )
