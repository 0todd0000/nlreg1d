
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import spm1d
import nlreg1d as nl



niter     = 5     # MaxItr for SRSF registration
penalty0  = 0.02  # elastic warping limit (case 1)
penalty1  = 0.01  # elastic warping limit (case 2)
save      = True



# load data:
dataset   = nl.data.SimulatedA()
y         = dataset.dv
group     = dataset.group
i0,i1     = group==0, group==1



# register:
J,Q       = y.shape
q         = np.linspace(0, 1, Q)
yr,wr     = nl.register_srsf( y, MaxItr=niter )
yre0,wre0 = nl.reg.elastic(y, q, penalty=penalty0)
yre1,wre1 = nl.reg.elastic(y, q, penalty=penalty1)
wlistr    = nl.Warp1DList( wr )
wlistre0  = nl.Warp1DList( wre0 )
wlistre1  = nl.Warp1DList( wre1 )
dr        = wlistr.get_displacement_field()[:,1:-1]
dre0      = wlistre0.get_displacement_field()[:,1:-1]
dre1      = wlistre1.get_displacement_field()[:,1:-1]



# plot:
plt.close('all')
fig,AX    = plt.subplots( 7, 2, figsize=(7,12) )
fontname  = 'Helvetica'
for i,yy in enumerate( [y,yr,yre0,yre1,dr,dre0,dre1] ):
	yy0,yy1 = yy[i0], yy[i1]
	h0 = AX[i,0].plot(yy0.T, 'k')[0]
	h1 = AX[i,0].plot(yy1.T, 'c')[0]
	ti  = spm1d.stats.ttest2( yy1, yy0 ).inference(0.05)
	ti.plot( ax=AX[i,1] )
	if i==0:
		leg = AX[i,0].legend( [h0,h1], ['Group 1', 'Group 2'] )
		plt.setp( leg.get_texts(), name=fontname, size=8)


for ax in AX.ravel():
	plt.setp( ax.get_xticklabels() + ax.get_yticklabels(), name=fontname, size=8 )
	ax.set_xlim(0, 100)
[ax.set_xticklabels([])  for ax in AX[:6].ravel()]
[ax.set_ylabel('SPM{t}', name=fontname, size=12)  for ax in AX[:,1]]


# column labels
labels = ['Dependent variable', 'Hypothesis testing results']
[ax.set_title(s, name=fontname, size=14) for ax,s in zip(AX[0], labels)]


# row group labels
labels = ['Original data', 'Nonlinearly registered data', 'Displacement fields']
for ax,s in zip(AX[[0,2,5],0], labels):
	ax.set_ylabel(s, size=16, name=fontname)
[ax.set_xlabel('Domain position (%)', name=fontname, size=16)  for ax in AX[-1]]


# panel labels
labels = [
	'Linearly registered',
	'Tucker et al. (2013)',
	f'Marron et al. (2015)\n   penalty = {penalty0}',
	f'Marron et al. (2015)\n   penalty = {penalty1}',
]
labels += labels[1:]
for i,(ax,s) in enumerate(zip(AX[:,0], labels)):
	ax.text( 0.03, 0.94, '(%s) %s' %(chr(97+i),s), size=11, name=fontname, transform=ax.transAxes, va='top')


# axis limits
plt.setp( AX[:4,0], ylim=(-5,35) )
plt.setp( AX[4:,0], ylim=(-0.2,0.48) )
plt.setp( AX[:,1], ylim=(-7,7) )

# background patch:
c0     = '0.8'
patch  = patches.Rectangle([0,0.445], 1, 0.40, facecolor=c0, edgecolor=c0, alpha=0.9, zorder=-1)
fig.add_artist(patch)


plt.tight_layout()
plt.show()







if save:
	plt.savefig( os.path.join(nl.dirFIGS, 'sensitivity.pdf') )


