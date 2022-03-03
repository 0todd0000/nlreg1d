

import numpy as np
from matplotlib import pyplot as plt
from spm1d import rft1d
import nlreg1d as nl



def tstat(y):  # one-sample t statistic
	return y.mean(axis=0) / y.std(ddof=1, axis=0) * (y.shape[0]**0.5)

def estimate_lkc(r):
	'''
	Estimate the Lipschitz-Killing curvature of a set of resoduals
	
	Reference:
	Taylor (2008) "Random fields of multivariate test statistics"
	'''
	def _norm(a):
		return (a**2).sum(axis=0)**0.5
	q   = np.diff( r / _norm(r).T , axis=1 )
	lkc = _norm( q ).sum()
	return lkc


def sim(nsim=200, shape_parameter=0.1, n_random=25, u=np.linspace(2, 5, 21)):
    tmax      = []     # maximum t-value
    fwhme     = []     # estimated FWHM
    lkce      = []     # estimate LKC
    for i in range(nsim):
        w     = nl.random_warp( J, Q, sigma=3, shape_parameter=shape_parameter, n_random=n_random, as_warp_object=True )
        # y     = w.get_deviation_from_linear_time()[:,1:-1]
        y     = w.get_displacement_field()[:,1:-1]
        r     = y - y.mean(axis=0)  # residuals
        t     = tstat( y )
        tmax.append( t.max() )
        fwhme.append( nl.util.estimate_fwhm(r) )
        lkce.append( estimate_lkc(r) )
    tmax      = np.array(tmax)
    # geometry summary:
    fwhmE     = np.mean( fwhme )
    lkcE      = np.mean( lkce )
    # survival functions:
    sf        = np.array( [ (tmax>uu).mean()  for uu in u] )  # simulation results
    sfe       = rft1d.t.sf(u, v, Q, fwhmE)           # expected
    return fwhmE, lkcE, sf,sfe




# set parameters:
np.random.seed(12345)
J            = 12      # number of observations
Q            = 101     # number of grid points
v            = J - 1   # degress of freedom
nsim         = 200     # number of simulated datasets (change this to 10000 to replicate the results in: ./Figures/sim-results.pdf)
u            = np.linspace(1, 5, 21)     # thresholds for survival function calculation
sps          = [10, 1, 0.1, 0.01]        # shape_parameter values
nrs          = [3,  6,  15,   50]        # n_random values


# run simulation:
fwhms,lkcs   = [],[]
sfs,sfes     = [],[]
for sp,nr in zip(sps,nrs):
    results  = sim(nsim, shape_parameter=sp, n_random=nr, u=u)
    fwhms.append( results[0] )
    lkcs.append( results[1] )
    sfs.append( results[2] )
    sfes.append( results[3] )



# plot:
plt.close('all')
fig,axs = plt.subplots( 2, 2, figsize=(8,6) )
plt.get_current_fig_manager().window.move(0, 0)
fontname = 'Helvetica'

colors = ['r', 'g', 'b', 'k']
for ax,sf,sfe,lkc,fwhm,c in zip(axs.ravel(), sfs, sfes, lkcs, fwhms, colors):
    h0 = ax.plot(u, sfe, '-', color=c, label='Theoretical')[0]
    h1 = ax.plot(u, sf,  'o', color=c, label='Simulated')[0]
    ax.text(0.6, 0.98, 'LKC = %.1f\nFWHM = %.1f'%(lkc,fwhm), transform=ax.transAxes, va='top', color=c, name=fontname)
    ax.set_ylim(0, 0.3)
    ax.grid(axis='x', color='0.9')
    ax.grid(axis='y', color='0.9')
    if ax==axs[0,1]:
        leg = ax.legend(loc='lower left')
        plt.setp( leg.get_texts(), name=fontname, size=10 )

[plt.setp(ax.get_xticklabels()+ax.get_yticklabels(), name=fontname, size=8)   for ax in axs.ravel()]
[ax.set_xticklabels([])  for ax in axs[0]]
[ax.set_yticklabels([])  for ax in axs[:,1]]
[ax.set_xlabel(r'$u$', name=fontname, size=14)  for ax in axs[1]]
[ax.set_ylabel(r'$P(t_\max > u)$', name=fontname, size=14)  for ax in axs[:,0]]
[ax.text(0.03, 0.92, '(%s)'%chr(97+i), name=fontname, size=12, transform=ax.transAxes)  for i,ax in enumerate(axs.ravel())]

plt.tight_layout()
plt.show()
