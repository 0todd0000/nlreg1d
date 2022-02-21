
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
from . import dirDATA




class _Dataset(object):
	
	fpath          = None   # path to data file
	
	def __init__(self):
		self.dv    = None   # dependent variable
		self.group = None   # group
		self._load()
		
	
	def __repr__(self):
		s   = f'Dataset: {self.name}\n'
		s  += f'    fpath   = {self.fpath}\n'
		s  += f'    shape   = {self.shape}\n'
		s  += f'    groups  = {self.ug.tolist()}\n'
		return s
	
	def _load(self):
		a          = np.loadtxt( self.fpath, delimiter=',')
		self.group = np.asarray(a[:,0], dtype=int)
		self.dv    = a[:,1:]
		
	@property
	def J(self):  # number of observations
		return self.dv.shape[0]

	@property
	def Q(self):  # number of grid points
		return self.dv.shape[1]

	@property
	def filename(self):  # dataset name
		return os.path.split( self.fpath )[1]

	@property
	def name(self):  # dataset name
		return self.__class__.__name__

	@property
	def q(self):  # grid points (equally spaced over [0,1])
		return np.linspace(0, 1, self.Q)

	@property
	def shape(self):  # dependent variable array shape
		return self.dv.shape

	@property
	def ug(self):  # unique group labels
		return np.unique(self.group)

	def get_dv_by_group(self):
		return [self.dv[self.group==u]  for u in self.ug]
	
	def plot(self, ax=None, colors=('b', 'r')):
		ax    = plt.gca() if (ax is None) else ax
		y0,y1 = self.get_dv_by_group()
		ax.plot(self.q, y0.T, color=colors[0], lw=0.3)
		ax.plot(self.q, y1.T, color=colors[1], lw=0.3)
		h0    = ax.plot(self.q, y0.mean(axis=0), color=colors[0], lw=5)[0]
		h1    = ax.plot(self.q, y1.mean(axis=0), color=colors[1], lw=5)[0]
		ax.legend([h0,h1], [f'Group {self.ug[0]} mean', f'Group {self.ug[1]} mean'])
		ax.set_title( self.name )


class Besier2009VastusForce(_Dataset):
	fpath = os.path.join( dirDATA, 'Besier2009-vastus.csv' )
	
class Dorn2012(_Dataset):
	fpath = os.path.join( dirDATA, 'Dorn2021-reduced.npz' )
	
	def _load(self):
		with np.load( self.fpath, allow_pickle=True ) as z:
			self.group = z['speed']
			self.dv    = z['y']

class Pataky2014MediolateralCOP(_Dataset):
	fpath = os.path.join( dirDATA, 'Pataky2014-mediolateral.csv' )

class SimulatedA(_Dataset):
	fpath = os.path.join( dirDATA, 'SimulatedA.csv' )
	
class SimulatedB(_Dataset):
	fpath = os.path.join( dirDATA, 'SimulatedB.csv' )


