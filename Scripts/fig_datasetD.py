
import os
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d as nl


# set parameters:
niter     = 5
wdv       = 'displacement'  # dependent variable for warp: 'deviation' or 'displacement'
save      = True



# load and register data:
dataset   = nl.data.Pataky2014MediolateralCOP()
y         = dataset.dv
J         = (dataset.group==0).sum()     # number of observations in first group
yr,wf     = nl.register_srsf(y, MaxItr=niter)
wlist     = nl.Warp1DList( wf )
d         = wlist.get_deviation_from_linear_time() if (wdv=='deviation') else wlist.get_displacement_field()



# plot:
plt.close('all')
colors    = '0.0', (0.8,0.1,0.1)
xlabel    = 'Time  (% stance)'
ylimt     = (-7.5, 7.5)
ylim      = [ (-2, 3), (-2, 3), (-0.5, 0.5), ylimt, (-2, 65), ylimt, ylimt ]
alpha_x   = [20, 20, 80, 80]
fig,AX    = nl.plot.plot_multipanel(y, yr, d, J, colors, ylim, alpha_x, paired=True, dvlabel='Mediolateral COP  (cm)', group_labels=['Normal', 'Fast'], xlabel=xlabel)
tx0,tx1   = AX[0].text(0.1, 0.8, 'Lateral'), AX[0].text(0.1, 0.1, 'Medial')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=AX[0].transAxes)
[ax.axhline(0, color='k', ls=':')  for ax in AX[:2]]
plt.show()
if save:
	plt.savefig( os.path.join(nl.dirFIGS, f'{dataset.name}.pdf')  )

