
import os,pathlib
from math import pi
import numpy as np
from matplotlib import pyplot as plt
import nlreg1d



# create single 1D observation to warp
Q        = 101
q        = np.linspace(0, 100, Q)
y        = np.sin( q * (2*pi) / (Q-1) )


# create random warps:
seeds    = [0, 1, 2]
sigmas   = [30, 20, 30]
w        = []
for sd,sg in zip(seeds,sigmas):
	np.random.seed(sd)
	ww   = nlreg1d.warp.random(Q, sg, J=1)
	w.append( ww )
wf       = [ww.dispf  for ww in w]     # warp (displacement) fields
yw       = [ww.apply(y)  for ww in w]  # warped data



# plot:
plt.close('all')
fig,AX = plt.subplots( 2, 2, figsize=(8,6) )
plt.get_current_fig_manager().window.move(0, 0)
ax0,ax1,ax2,ax3 = AX.ravel()

fontname = 'Helvetica'
colors   = [ [0.733, 0.576, 0.761],
             [0.588, 0.702, 0.867],
             [0.494, 0.808, 0.957] ]

### plot warps:
ax0.plot(q, q, 'k:')
[ax0.plot( q, 100*ww.asarray(), color=cc)  for ww,cc in zip(w,colors)]

### plot displacement fields:
ax1.axhline(0, color='k', ls=':')
[ax1.plot( q, 100*wwf, color=cc)  for wwf,cc in zip(wf,colors)]

### plot reference data:
ax2.plot(q, y, 'k', lw=2)

### plot warped data:
ax3.plot(q, y, 'k:')
[ax3.plot(q, yyw, color=cc, lw=3)  for (yyw,cc) in zip(yw,colors)]
for ax in [ax2,ax3]:
	[ax.axhline(yy, color='k', ls=':', lw=0.5, zorder=-1)  for yy in [-1,0,1]]
	# ax.axhline(0, color='k', ls=':', lw=0.5)
	# [ax.axvline(qq, color='k', ls=':', lw=0.5, zorder=-1)  for qq in [25,50,75]]

### add reference vertical lines
for ax in AX.ravel():
	[ax.axvline(qq, color='k', ls=':', lw=0.5, zorder=-1)  for qq in [25,50,75]]



### limits:
ax1.set_ylim(-12, 29.5)
[ax.set_ylim(-1.3, 1.3)  for ax in AX[1]]

### ticks:
[plt.setp(ax.get_xticklabels()+ax.get_yticklabels(), name=fontname, size=8)   for ax in AX.ravel()]
[ax.set_yticks( np.linspace(-1,1,5) )  for ax in AX[1]]
[ax.set_xticklabels([])  for ax in AX[0]]

### axis labels:
[ax.set_xlabel('Time  (%)', name=fontname, size=12)  for ax in AX.ravel()]
ylabels = ['Time  (%)  [warped]', 'Temporal shift  (%)', 'Dependent variable value', 'Dependent variable value']
[ax.set_ylabel(ss, name=fontname, size=12)  for ax,ss in zip(AX.ravel(), ylabels)]

### legend:
leg = ax0.legend(['Null warp', 'Warp 1', 'Warp 2', 'Warp 3'], loc='lower right')
plt.setp( leg.get_texts(), name=fontname, size=8)

### panel labels:
labels = ['Warp functions', 'Displacement fields', 'Reference data', 'Warped data']
[ax.text(0.03, 0.93, '(%s)  %s' %(chr(97+i).upper(), ss), name=fontname, size=12, transform=ax.transAxes)   for i,(ax,ss) in enumerate( zip(AX.ravel(), labels) )   ]


plt.tight_layout()

plt.show()


dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
plt.savefig( os.path.join(dirFIGS, 'warp.pdf')  )





