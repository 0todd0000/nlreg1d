
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = False



# Dataset C:  vastus forces (Besier et al., 2009):
fnameCSV  = os.path.join( dirDATA, 'Besier2009-vastus.csv')
a         = np.loadtxt( fnameCSV, delimiter=',')
g,y       = a[:,0], a[:,1:]  # group, dependent variable
J         = (g==0).sum()     # number of observations in first group
### register:
Q         = y.shape[1]
q         = np.linspace(0, 1, Q)
# yr,wr     = nlreg1d.reg.fpca( y, ncomp=5, smooth=False, niter=niter )
yr,wr     = nlreg1d.reg.srsf(y, MaxItr=5)
### plot:
plt.close('all')
colors    = '0.0', (0.8,0.1,0.1)
xlabel    = 'Time  (% stance)'
# ylim      = [ (-0.05, 1.3), (-0.05, 1.3), (-25, 39), (-6, 6), (-2, 30), (-6,6), (-6,6) ]
ylim      = [ (-0.05, 1.3), (-0.05, 1.3), (-0.5, 0.5), (-6, 6), (-2, 30), (-6,6), (-6,6) ]
alpha_x   = [70, 70, 70, 70]
leg_loc   = [(0.80, 0.92), (0.80, 0.92), (0.99, 0.99)]
fig,AX    = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_x, paired=False, dvlabel='Muscle force  (kN)', group_labels=['Control mean', 'PFP mean'], xlabel=xlabel, leg_loc=leg_loc)
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'datasetC.pdf')  )

