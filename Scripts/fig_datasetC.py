
import os
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d as nl


# set parameters:
niter     = 5
wdv       = 'displacement'  # dependent variable for warp: 'deviation' or 'displacement'
save      = True



# load and register data:
dataset   = nl.data.Besier2009VastusForce()
y         = dataset.dv
J         = (dataset.group==0).sum()     # number of observations in first group
yr,wf     = nl.register_srsf(y, MaxItr=niter)
wlist     = nl.Warp1DList( wf )
d         = wlist.get_deviation_from_linear_time() if (wdv=='deviation') else wlist.get_displacement_field()



# plot:
plt.close('all')
colors    = '0.0', (0.8,0.1,0.1)
xlabel    = 'Time  (% stance)'
ylim      = [ (-0.05, 1.3), (-0.05, 1.3), (-0.5, 0.5), (-6, 6), (-2, 30), (-6,6), (-6,6) ]
alpha_x   = [70, 70, 70, 70]
leg_loc   = [(0.80, 0.92), (0.80, 0.92), (0.99, 0.99)]
nl.plot.plot_multipanel(y, yr, d, J, colors, ylim, alpha_x, paired=False, dvlabel='Muscle force  (kN)', group_labels=['Control mean', 'PFP mean'], xlabel=xlabel, leg_loc=leg_loc)
plt.show()
if save:
	plt.savefig( os.path.join(nl.dirFIGS, f'{dataset.name}.pdf')  )
