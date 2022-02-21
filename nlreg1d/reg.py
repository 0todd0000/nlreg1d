

import os,sys
# import collections
import numpy as np
from scipy import interpolate
# from numpy.linalg import norm
# from scipy.linalg import svd
# from scipy.integrate import trapz, cumtrapz
import fdasrsf
# import fdasrsf.utility_functions as uf
import skfda




def _disable_print():
	sys.stdout = open(os.devnull, 'w')

def _enable_print():
	sys.stdout = sys.__stdout__


def srsf(y, verbose=True, **kwargs):
	# method="mean", omethod="DP2", center=True, smoothdata=False, MaxItr=5, parallel=False, lam=0.0, cores=-1, grid_dim=7, parallel=False, verbose=False
	'''
	Conduct nonlinear registration on a set of 1D univariate observations
	using the SRSF (square-root slope) framework
	
	*** This function is a wrapper for fdasrsf.fdawarp.srsf_align ***
	
	See:
	
		https://github.com/jdtuck/fdasrsf_python/blob/master/fdasrsf/time_warping.py


	Usage:
	
	>>> import numpy as np
	>>> import nlreg1d as nl
	>>> y     = np.random.randn(8, 101)  # 8 observations, 101 nodes each
	>>> yr,wf = nl.reg.srsf( y )
	>>> yr,wf = nl.reg.srsf( y, verbose=False, method="mean",
				omethod="DP2", center=True, smoothdata=False,
				MaxItr=20, parallel=False, lam=0.0, 
				cores=-1, grid_dim=7 )


	Inputs:
	
	*y* : (J,Q) data array
	
	where:
		J = number of observations
		Q = number of domain nodes
	
	*Keyword arguments*: (see below)
	
	
	Outputs:
	
	*yr* : (J,Q) array --- nonlinearly registered data 
	
	*wr* : (J,Q) array --- optimal warping functions
	
	

	(The following keyword argument description is modified from: time_warping.py/fdawarp/srsf_align)
	
	Keyword arguments:

	*method*      ("mean" or "median") (default = "mean")
	
	*omethod*     ("DP", "DP2" or "RBFGS") (default = "DP2")
	
	*center*      center warping functions (default = True)
	
	*smoothdata*  smooth using a box filter (default = Fals)
	
	*MaxItr*      maximum number of iterations (default = 20)
	
	*parallel*    run in parallel (default = False)
	
	*lam*         elasticity control (default = 0)
	
	*cores*       number of cores for parallel (default = -1 (all))
	
	*grid_dim*    size of the grid, for the DP2 method only (default = 7)
	'''
	Q    = y.shape[1]               # number of points
	q    = np.linspace(0, 1, Q)     # grid points
	fdaw = fdasrsf.fdawarp(y.T, q)  # warping object
	if not verbose:
		_disable_print()
	fdaw.srsf_align(**kwargs)       # conduct registration
	_enable_print()
	yr   = fdaw.fn.T
	wf   = fdaw.gam.T
	return yr, wf






def elastic(y, q=None, penalty=0):
	'''
	Wrapper for skfda.preprocessing.registration.ElasticRegistration
	
	This is an alternative implementation of SRSF-based registration.
	
	This function is not used in the main manuscript's analyses; it
	appears only for development purposes
	'''
	Q     = y.shape[1]
	if q is None:
	    q = np.linspace(0, 1, Q)
	fd    = skfda.FDataGrid( data_matrix=y, grid_points=q)
	er    = skfda.preprocessing.registration.ElasticRegistration(penalty=penalty)
	fdr   = er.fit_transform(fd)
	yr    = fdr.data_matrix[:,:,0]
	wf    = er.warping_.data_matrix[:,:,0]
	return yr,wf
	
	
def linear(y, n=101):
	'''
	Linearly register (interpolate) a single (Q,) observation to *n* equally spaced points
	'''
	Q     = y.size
	q0    = np.linspace(0, 1, Q )
	qi    = np.linspace( q0.min(), q0.max(), n )
	f     = interpolate.interp1d(q0, y)
	yi    = f(qi)
	return yi