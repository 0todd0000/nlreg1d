
import os
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d as nl






niter     = 5
save      = True





# Dorn 2012 dataset
fnameNPZ  = os.path.join( nl.dirDATA, 'Dorn2021-orig.npz')
with np.load( fnameNPZ, allow_pickle=True ) as Z:
	Y     = Z['Y']
	SPEED = Z['SPEED']
	FOOT  = Z['FOOT']
i         = FOOT==0
Y,SPEED   = Y[i], SPEED[i]
y         = [yy[:,0]  for yy in Y]


# register (linear):
yi        = np.array( [nl.register_linear(yy, n=101)   for yy in y] )


# register (nonlinear):
yr,wf     = nl.register_srsf(yi, MaxItr=niter)
wlist     = nl.Warp1DList(wf)
d         = wlist.get_displacement_field()




# plot:
plt.close('all')
fig,AX = plt.subplots( 2, 2, figsize=(8,6) )
ax0,ax1,ax2,ax3 = AX.ravel()

fontname = 'Helvetica'
colors   = np.array([
		[0.580, 0.792, 0.510],
		[0.494, 0.808, 0.957],
		[0.733, 0.576, 0.761],
		[0.300, 0.300, 0.300]   ])
c0,c1,c2,c3 = colors
colors      = [c0, c0, c1, c1, c2, c2, c3, c3]



[ax0.plot(yy, color=cc)  for yy,cc in zip(y,colors)]
[ax1.plot(yy, color=cc)  for yy,cc in zip(yi,colors)]
[ax2.plot(yy, color=cc)  for yy,cc in zip(yr,colors)]
[ax3.plot(yy, color=cc)  for yy,cc in zip(d,colors)]
[ax.axhline(0, color='0.7', ls=':')  for ax in AX.ravel()]

# add legend:
uspeed = np.unique(SPEED)
labels = [f'{x} m/s'  for x in uspeed]
lines  = np.array( ax0.lines )[[0,2,4,6]]
leg    = ax0.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.4,0.03))
plt.setp( leg.get_texts(), name=fontname, size=8)

### limits:
[ax.set_ylim(-690, 690)  for ax in [ax0,ax1,ax2]]
ylim   = -0.1, 0.1
ax3.set_ylim(*ylim)
ax3.set_yticks( np.linspace(ylim[0], ylim[1], 5) )

### direction labels
tx0 = ax0.text(0.88, 0.75, 'Anterior')
tx1 = ax0.text(0.88, 0.25, 'Posterior')
plt.setp( [tx0,tx1], name='Helvetica', size=10, bbox=dict(facecolor='0.9'), transform=ax0.transAxes, ha='center')


### ticks:
[plt.setp(ax.get_xticklabels()+ax.get_yticklabels(), name=fontname, size=8)   for ax in AX.ravel()]

### axis labels:
ax0.set_xlabel('Time  (ms)', name=fontname, size=12)
[ax.set_xlabel('Time  (%)', name=fontname, size=12)  for ax in AX.ravel()[1:]]
ylabels = ['GRF  (N)'] * 3 + ['Temporal displacement (%)']
[ax.set_ylabel(ss, name=fontname, size=12)  for ax,ss in zip(AX.ravel(), ylabels)]


### panel labels:
labels = ['Original', 'Linearly registered', 'Nonlinearly registered', 'Displacement fields']
[ax.text(0.03, 0.93, '(%s)  %s' %(chr(97+i).upper(), ss), name=fontname, size=12, transform=ax.transAxes)   for i,(ax,ss) in enumerate( zip(AX.ravel(), labels) )]


plt.tight_layout()

plt.show()


if save:
	plt.savefig( os.path.join(nl.dirFIGS, 'example.pdf')  )





