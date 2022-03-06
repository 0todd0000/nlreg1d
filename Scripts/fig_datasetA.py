
import os
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d as nl


# set parameters:
np.random.seed(123456789)
niter      = 5      # max iterations for SRSF registration
save       = True   # save figure as PDF?
parametric = False  # parametric or nonparametric inference
nperm      = 1000   # number of permutations for SnPM ("Fmax") inference



# load and register data:
dataset    = nl.data.SimulatedA()
y          = dataset.dv
J          = (dataset.group==0).sum()     # number of observations in first group
yr,wf      = nl.register_srsf(y, MaxItr=niter)
wlist      = nl.Warp1DList( wf )
d          = wlist.get_displacement_field()



# plot:
plt.close('all')
colors    = '0.0', (0.3,0.5,0.99)
ylimt     = (-7.5, 7.5)
ylim      = [ (-5, 35), (-5, 35), (-0.19, 0.59), ylimt, (-2, 60), ylimt, ylimt ]
alpha_x   = [70, 70, 70, 70]
leg_loc   = [(0.99, 0.92), (0.99, 0.92), (0.50, 0.92)]
nl.plot.plot_multipanel(y, yr, d, J, colors, parametric, ylim, alpha_x, paired=False, permutations=nperm, dvlabel='Dependent variable value', leg_loc=leg_loc)
plt.show()
if save:
	plt.savefig( os.path.join(nl.dirFIGS, f'{dataset.name}.pdf')  )

