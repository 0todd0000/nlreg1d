
'''
Estimate LKC for all datasets
'''

import numpy as np
import matplotlib.pyplot as plt
import nlreg1d as nl



ds0   = nl.data.SimulatedA()
ds1   = nl.data.SimulatedB()
ds2   = nl.data.Besier2009VastusForce()
ds3   = nl.data.Pataky2014MediolateralCOP()
dss   = [ds0, ds1, ds2, ds3]

dfr   = []
lkc   = []
for ds in dss:
	y         = ds.dv
	yr,wf     = nl.register_srsf(y, MaxItr=5)
	wlist     = nl.Warp1DList( wf )
	df        = wlist.get_displacement_field()[:,1:-1]
	r         = df - df.mean(axis=0)
	dfr.append( r )
	lkc.append( nl.util.estimate_lkc( r ) )
	# lkc.append( nl.util.estimate_fwhm( r ) )
	print( f'%s:  %.3f' %(ds.name, lkc[-1]))


plt.close('all')
fig,axs = plt.subplots( 2, 2, figsize=(8,6) )
for ax,ds,df,x in zip(axs.ravel(), dss, dfr, lkc):
	ax.plot(df.T, 'k-', lw=0.5)
	ax.text(0.05, 0.92, '%s: LKC=%.3f' %(ds.name, x), transform=ax.transAxes)
	ax.set_ylim(-0.4, 0.4)
plt.tight_layout()
plt.show()


