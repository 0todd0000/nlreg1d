
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import skfda
import fdasrsf


class _WarpBase(object):
	
	@property
	def dev(self):  # deviations from linear time (warped grid)
		return self.w - self.q0

	def _gca(self, ax):
		return plt.gca() if (ax is None) else ax
	
	def asarray(self):
		return self.w.copy()



	def get_grid_points(self, relative=True):
		return self.q0 if relative else (self.Q * self.q0)

	def get_deviation_from_linear_time(self, relative=True):  # relative to domain [0,1] if True, otherwise [0,Q]
		return self.dev if relative else (self.Q * self.dev)
	
	def get_displacement_field(self, relative=True):
		return self.dispf if relative else (self.Q * self.dispf)

	def get_warp_function(self, relative=True):
		return self.w if relative else (self.Q * self.w)

	def get_inverse(self, as_warp_object=False):
		return self.__class__(self.inv) if as_warp_object else self.inv



	def plot(self, ax=None, relative=True, **kwargs):
		self.plot_warp_function( ax=ax, relative=relative, **kwargs )

	def plot_deviation_from_linear_time(self, ax=None, relative=True, **kwargs):
		ax = self._gca(ax)
		x  = self.get_grid_points( relative=relative )
		y  = self.get_deviation_from_linear_time( relative=relative )
		ax.plot( x, y.T, **kwargs)

	def plot_displacement_field(self, ax=None, relative=True, **kwargs):
		ax = self._gca(ax)
		x  = self.get_grid_points( relative=relative )
		y  = self.get_displacement_field( relative=relative )
		ax.plot( x, y.T, **kwargs)

	def plot_warp_function(self, ax=None, relative=True, **kwargs):
		ax = self._gca(ax)
		x  = self.get_grid_points( relative=relative )
		y  = self.get_warp_function( relative=relative )
		ax.plot( x, y.T, **kwargs)





class Warp1D(_WarpBase):
	def __init__(self, w):
		self.w    = np.asarray(w, dtype=float) # warp function
		self.q0   = np.linspace(0, 1, self.Q)  # grid points

	@property
	def Q(self):  # number of grid points
		return self.w.size

	@property
	def dispf(self):  # displacement field (original grid)
		x,y   = self.qw, self.dev
		f     = interpolate.interp1d( x, y, 'linear', bounds_error=False, fill_value=0)
		return -f( self.q0 )
		
	@property
	def inv(self):  # inverse warp
		f = interpolate.interp1d( self.w, self.q0, 'linear', bounds_error=False, fill_value=0)
		w = f( self.q0 )
		return w

	@property
	def qw(self):  # warped grid
		return self.apply( self.q0 )

	@property
	def size(self):  # number of grid points
		return self.w.size

	def apply(self, y):  # apply warp to a (Q,) array
		f     = interpolate.interp1d( self.q0, y, 'linear', bounds_error=False, fill_value=0)
		return f( self.w )
		





class Warp1DList(list, _WarpBase):
	def __init__(self, w):
		self.w    = np.asarray(w, dtype=float)
		self.q0   = np.linspace(0, 1, self.Q)
		super().__init__( [Warp1D(ww)  for ww in w] )
		
	@property
	def J(self):  # number of observations
		return self.w.shape[0]

	@property
	def Q(self):  # number of grid points
		return self.w.shape[1]

	@property
	def dispf(self):  # displacement fields
		return np.array([w.dispf   for w in self])
		
	@property
	def inv(self):  # inverse warps
		return np.array( [ww.inv for ww in self] )

	@property
	def shape(self):
		return self.w.shape
		
	def apply(self, y):   # apply warps to (J,Q) array of observations
		return np.array( [ ww.apply(yy)   for ww,yy in zip(self, y)] )
	


def random_warp(J=1, Q=101, sigma=5, shape_parameter=2, n_random=5, as_warp_object=False):
	'''
	Wrapper for skfda.datasets.make_random_warping
	'''
	fgrid  = skfda.datasets.make_random_warping(n_samples=J, n_features=Q, start=0, stop=1, sigma=sigma, shape_parameter=shape_parameter, n_random=n_random, random_state=None)
	q      = fgrid.grid_points[0]
	w      = fgrid.data_matrix.squeeze()
	if as_warp_object:
		w  = Warp1D( w ) if (J==1) else Warp1DList( w )
	return w


