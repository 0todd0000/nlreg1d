
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import skfda
import fdasrsf
import fdasrsf.utility_functions as uf



class Warp1D(object):
	def __init__(self, w):
		self.w    = w
		self.q0   = np.linspace(0, 1, self.Q)

	@property
	def Q(self):
		return self.w.size
	@property
	def dev(self):
		'Linear time deviations (differences from linear time)'
		return self.w - self.q0

	@property
	def dispf(self):
		'Displacement field (original coordinates)'
		x,y   = self.qw, -self.wfn
		f     = interpolate.interp1d( x, y, 'linear', bounds_error=False, fill_value=0)
		return f( self.q0 )
		
	@property
	def inv(self):
		'Inverse warp'
		f     = interpolate.interp1d( self.w, self.q0, 'linear', bounds_error=False, fill_value=0)
		return self.__class__(  f( self.q0 )  )
	@property
	def qw(self):
		'Warped time field'
		return self.apply( self.q0 )
	@property
	def wfn(self):
		'Warp function'
		return -(self.w - self.q0)
	
	def _gca(self, ax):
		return plt.gca() if (ax is None) else ax
	
	def apply(self, y, interp='linear'):
		# same as:   uf.warp_f_gamma( self.q0, y, self.w )
		# qw    = self.w
		f     = interpolate.interp1d( self.q0, y, interp, bounds_error=False, fill_value=0)
		return f( self.w )
		
		
		
		# x0    = self.get_original_domain()
		# xw    = self.get_warped_domain()
		# f     = interpolate.interp1d(xw, y, 'linear', bounds_error=False, fill_value=0)
		# return f(x0)
		
		
	def asarray(self, df=False):
		return self.df if df else self.w
	
	def get_displacement_field(self, interp='linear', rel=True):
		x,y   = self.qw, -self.wfn
		f     = interpolate.interp1d( x, y, interp, bounds_error=False, fill_value=0)
		df    = - f( self.q0 )
		if not rel:
			df *= self.Q
			# df = Q * df
		#
		#
		# yi    = f( self.q0 )
		# df    = - np.array( yi )
		# if not rel:
		# 	Q  = x.size
		# 	df = Q * df
		return df
	
	def get_inverse(self):
		return self.inv
		
	def get_deviation_from_linear_time(self, rel=True):
		d = self.dev
		if not rel:
			d *= self.Q
		return d
	
	
	def plot(self, ax=None, **kwargs):
		ax = self._gca(ax)
		ax.plot( self.w, **kwargs)

	def plot_displacement_field(self, ax=None, **kwargs):
		ax = self._gca(ax)
		ax.plot( self.df, **kwargs)
		
		
	def verify(self):
		# (that it is a valid warp;  requiring monotonically increasing values = positive derivatives)
		pass



class Warp1DList(list):
	def __init__(self, w):
		self.w    = w
		self.q0   = np.linspace(0, 1, self.Q)
		super().__init__( [Warp1D(ww)  for ww in w] )
		
	@property
	def J(self):
		return self.w.shape[0]
	@property
	def Q(self):
		return self.w.shape[1]
	@property
	def wfn(self):
		return -(self.w - self.q0)
	@property
	def dispf(self):
		return np.array([w.dispf   for w in self])
		# return np.array([w.df   for w in self])
	@property
	def inv(self):
		wi = np.array([ww.inv.w for ww in self])
		return self.__class__(  wi  )
	@property
	def shape(self):
		return self.w.shape
		
	def _gca(self, ax):
		return plt.gca() if (ax is None) else ax
	
	def apply(self, y):
		return np.array( [ ww.apply(yy)   for ww,yy in zip(self, y)] )
	
	def asarray(self):
		return self.w.copy()

	def get_displacement_fields(self, interp='linear', rel=True):
		return np.array( [ ww.get_displacement_field(interp=interp, rel=rel)   for ww in self] )
	
	def get_deviations_from_linear_time(self):
		return np.array( [ww.dev  for ww in self] )
	
	def plot(self, ax=None, **kwargs):
		ax = self._gca(ax)
		ax.plot( self.w.T, **kwargs)

	def plot_displacement_field(self, ax=None, **kwargs):
		ax = self._gca(ax)
		ax.plot( self.df.T, **kwargs)
		

# def random(Q, sigma, J=1):
# 	w = uf.rgam(Q, sigma, J)
# 	if J==1:
# 		w = Warp1D( w.flatten() )
# 	else:
# 		w = Warp1DList( w.T )
# 	return w


def random(J=8, Q=101, sigma=5, shape_parameter=2, n_random=8, aswarp1d=False):
	fgrid = skfda.datasets.make_random_warping(n_samples=J, n_features=Q, start=0, stop=1, sigma=sigma, shape_parameter=shape_parameter, n_random=n_random, random_state=None)
	q     = fgrid.grid_points[0]
	wf    = fgrid.data_matrix.squeeze()
	if aswarp1d:
		wf = Warp1DList( wf )
	return wf
	# wlist = nl.warp.
	#
	# w = uf.rgam(Q, sigma, J)
	# if J==1:
	# 	w = Warp1D( w.flatten() )
	# else:
	# 	w = Warp1DList( w.T )
	# return w

