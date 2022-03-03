
from math import log
import numpy as np


def estimate_fwhm(R):
	eps    = np.finfo(float).eps
	ssq    = (R**2).sum(axis=0)
	dy,dx  = np.gradient(R)  # gradient estimate
	v      = (dx**2).sum(axis=0)
	v     /= (ssq + eps)  # normalized gradient estimate
	v      = v[np.logical_not( np.isnan(v) )] # ignore zero-variance nodes:
	rpn    = np.sqrt(v / (4*log(2)))   # global FWHM estimate (resels per node)
	fwhme  = (1.0 / rpn).mean()   # estimated FWHM
	return fwhme


def estimate_lkc(r):
	'''
	Estimate the Lipschitz-Killing curvature of a set of residuals
	
	Reference:
	Taylor (2008) "Random fields of multivariate test statistics"
	'''
	def _norm(a):
		return (a**2).sum(axis=0)**0.5
	q   = np.diff( r / _norm(r).T , axis=1 )
	lkc = _norm( q ).sum()
	return lkc