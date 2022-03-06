
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
dataset   = nl.data.Pataky2014MediolateralCOP()
y         = dataset.dv
J         = (dataset.group==0).sum()     # number of observations in first group
yr,wf     = nl.register_srsf(y, MaxItr=niter)
wlist     = nl.Warp1DList( wf )
d         = wlist.get_displacement_field()



# plot:
plt.close('all')
colors    = '0.0', (0.8,0.1,0.1)
xlabel    = 'Time  (% stance)'
ylimt     = (-7.5, 7.5)
ylim      = [ (-2, 3), (-2, 3), (-0.5, 0.5), ylimt, (-2, 90), ylimt, ylimt ]
alpha_x   = [20, 20, 80, 80]
leg_loc   = [(0.80, 0.29), (0.80, 0.29), (0.50, 0.29)]
fig,AX    = nl.plot.plot_multipanel(y, yr, d, J, colors, parametric, ylim, alpha_x, paired=True, permutations=nperm, dvlabel='Mediolateral COP  (cm)', group_labels=['Normal', 'Fast'], xlabel=xlabel, leg_loc=leg_loc)
tx0,tx1   = AX[0].text(0.1, 0.8, 'Lateral'), AX[0].text(0.1, 0.1, 'Medial')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=AX[0].transAxes)
[ax.axhline(0, color='k', ls=':')  for ax in AX[:2]]
plt.show()
if save:
	plt.savefig( os.path.join(nl.dirFIGS, f'{dataset.name}.pdf')  )

