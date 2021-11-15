
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import spm1d
import nlreg1d



dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
penalty0  = 0.02  # elastic warping limit
penalty1  = 0.01  # elastic warping limit
save      = True



# Dataset A (simulated)
fnameCSV  = os.path.join( dirDATA, 'SimulatedA.csv')
a         = np.loadtxt( fnameCSV, delimiter=',')
g,y       = a[:,0], a[:,1:]  # group, dependent variable
J         = (g==0).sum()     # number of observations in first group



### register:
Q         = y.shape[1]
q         = np.linspace(0, 1, Q)
yr,wr     = nlreg1d.reg.fpca( y, ncomp=5, smooth=False, niter=niter )
yre0,wre0 = nlreg1d.reg.elastic(y, q, penalty=penalty0)
yre1,wre1 = nlreg1d.reg.elastic(y, q, penalty=penalty1)
wlistr    = nlreg1d.warp.Warp1DList( wr )
wlistre0  = nlreg1d.warp.Warp1DList( wre0 )
wlistre1  = nlreg1d.warp.Warp1DList( wre1 )
dr        = wlistr.dispf[:,1:-1]
dre0      = wlistre0.dispf[:,1:-1]
dre1      = wlistre1.dispf[:,1:-1]

# plot:
plt.close('all')
fig,AX    = plt.subplots( 7, 2, figsize=(7,12) )
fontname  = 'Helvetica'
for i,yy in enumerate( [y,yr,yre0,yre1,dr,dre0,dre1] ):
	yy0,yy1 = yy[:J], yy[J:]
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


labels = ['Original data', 'Nonlinearly registered data', 'Warp fields']
for ax,s in zip(AX[[0,2,5],0], labels):
	ax.set_ylabel(s, size=16, name=fontname)
[ax.set_xlabel('Domain position (%)', name=fontname, size=16)  for ax in AX[-1]]


labels = [
	'Linearly registered',
	'Tucker et al. (2013)',
	f'Marron et al. (2015)\n   penalty = {penalty0}',
	f'Marron et al. (2015)\n   penalty = {penalty1}',
]
labels += labels[1:]
for i,(ax,s) in enumerate(zip(AX[:,0], labels)):
	ax.text( 0.03, 0.94, '(%s) %s' %(chr(97+i),s), size=11, name=fontname, transform=ax.transAxes, va='top')


plt.setp( AX[:4,0], ylim=(-5,35) )
plt.setp( AX[4:,0], ylim=(-20,40) )
plt.setp( AX[:,1], ylim=(-7,7) )

# background panels:
c0     = '0.8'
patch  = patches.Rectangle([0,0.435], 1, 0.42, facecolor=c0, edgecolor=c0, alpha=0.9, zorder=-1)
fig.add_artist(patch)


plt.tight_layout()
plt.show()







if save:
	dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
	plt.savefig( os.path.join(dirFIGS, 'supp_datasetA_warp_algorithm.pdf')  )


