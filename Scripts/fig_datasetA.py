
import os
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d as nl


# set parameters:
niter     = 5
wdv       = 'displacement'  # dependent variable for warp: 'deviation' or 'displacement'
save      = False



# load and register data:
dataset   = nl.data.SimulatedA()
y         = dataset.dv
J         = (dataset.group==0).sum()     # number of observations in first group
yr,wf     = nl.register_srsf(y, MaxItr=niter)
wlist     = nl.Warp1DList( wf )
d         = wlist.get_deviation_from_linear_time() if (wdv=='deviation') else wlist.get_displacement_field()



# plot:
plt.close('all')
colors    = '0.0', (0.3,0.5,0.99)
ylimt     = (-7.5, 7.5)
ylim      = [ (-5, 35), (-5, 35), (-0.19, 0.59), ylimt, (-2, 60), ylimt, ylimt ]
alpha_x   = [70, 70, 70, 70]
nl.plot.plot_multipanel(y, yr, d, J, colors, ylim, alpha_x, paired=False, dvlabel='Dependent variable value')
plt.show()
if save:
	plt.savefig( os.path.join(nl.dirFIGS, f'{dataset.name}.pdf')  )

