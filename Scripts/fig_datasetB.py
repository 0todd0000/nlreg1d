
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = True



# Dataset B (simulated)
fnameCSV  = os.path.join( dirDATA, 'SimulatedB.csv')
a         = np.loadtxt( fnameCSV, delimiter=',')
g,y       = a[:,0], a[:,1:]  # group, dependent variable
J         = (g==0).sum()     # number of observations in first group
### register:
Q         = y.shape[1]
q         = np.linspace(0, 1, Q)
yr,wr     = nlreg1d.reg.fpca( y, ncomp=5, smooth=False, niter=niter )
### plot:
plt.close('all')
colors    = '0.0', (0.3,0.5,0.99)
ylimt     = (-7, 7)
ylim      = [ (-5, 30), (-5, 30), (-40, 40), ylimt, (-2, 50), ylimt, ylimt ]
alpha_x   = [70, 70, 70, 70]
nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_x, paired=False, dvlabel='Dependent variable value')
plt.show()
if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'datasetB.pdf')  )
