



import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import spm1d



def axes2data(ax, points):
	ax.get_xlim()
	ax.get_ylim()
	t = ax.transAxes + ax.transData.inverted()
	return t.transform( points )

def data2axes(ax, points):
	ax.get_xlim()
	ax.get_ylim()
	t = (ax.transAxes + ax.transData.inverted()).inverted()
	return t.transform( points )





def plot_multipanel(y, yr, d, n0, colors, ylim=None, alpha_x=None, paired=False, dvlabel='Dependent variable', xlabel='Domain position  (%)', group_labels=None, leg_loc=[(0.99, 0.92), (0.99, 0.92), (0.99, 0.99)]):
	d        = d[:,1:-1]
	Y        = np.dstack( [yr[:,1:-1],d] )
	J        = n0
	fontname = 'Helvetica'
	glabels  = ['Group 1 mean', 'Group 2 mean'] if (group_labels is None) else group_labels
	
	# stats:
	if paired:
		ti      = spm1d.stats.ttest_paired( y[J:], y[:J] ).inference(0.05)
		T2i     = spm1d.stats.hotellings_paired( Y[J:], Y[:J] ).inference(0.05)
		tri     = spm1d.stats.ttest_paired( yr[J:], yr[:J] ).inference(0.05/2)
		twi     = spm1d.stats.ttest_paired( d[J:], d[:J] ).inference(0.05/2)
	else:
		ti      = spm1d.stats.ttest2( y[J:], y[:J] ).inference(0.05)
		T2i     = spm1d.stats.hotellings2( Y[J:], Y[:J] ).inference(0.05)
		tri     = spm1d.stats.ttest2( yr[J:], yr[:J] ).inference(0.05/2)
		twi     = spm1d.stats.ttest2( d[J:], d[:J] ).inference(0.05/2)
	
	
	# create figure and axes:
	fig         = plt.figure(  figsize=(14,10)  )
	axw,axh     = 0.26, 0.27
	# axx     = np.linspace( 0.06, 0.42, 0.71)
	axx         = [0.085, 0.415, 0.730]
	axy         = np.linspace(0.66, 0.06, 3)
	ax0,ax1,ax2 = [plt.axes( [x,axy[0],axw,axh] )  for x in axx]
	ax3         = plt.axes( [axx[0],axy[2],axw,axh] )
	ax4         = plt.axes( [axx[1]+0.15,axy[1],axw,axh] )
	ax5,ax6     = [plt.axes( [x,axy[2],axw,axh] )  for x in axx[1:]]
	AX          = [ax0,ax1,ax2, ax3, ax4, ax5,ax6]

	h0  = ax0.plot( y[:J].T, color=colors[0], lw=0.3 )[0]
	h1  = ax0.plot( y[J:].T, color=colors[1], lw=0.3 )[0]
	h0  = ax0.plot( y[:J].mean(axis=0), color=colors[0], lw=5 )[0]
	h1  = ax0.plot( y[J:].mean(axis=0), color=colors[1], lw=5 )[0]
	leg = ax0.legend( [h0,h1], glabels, loc='upper right', bbox_to_anchor=leg_loc[0] )
	plt.setp( leg.get_texts(), name=fontname)


	ax1.plot( yr[:J].T, color=colors[0], lw=0.3 )
	ax1.plot( yr[J:].T, color=colors[1], lw=0.3 )
	h0  = ax1.plot( yr[:J].mean(axis=0), color=colors[0], lw=5 )[0]
	h1  = ax1.plot( yr[J:].mean(axis=0), color=colors[1], lw=5 )[0]
	leg = ax1.legend( [h0,h1], glabels, loc='upper right', bbox_to_anchor=leg_loc[1]  )
	plt.setp( leg.get_texts(), name=fontname)


	h0 = ax2.plot( d[:J].T, color=colors[0], lw=0.3 )[0]
	h1 = ax2.plot( d[J:].T, color=colors[1], lw=0.3 )[0]
	h2 = ax2.axhline(0, color='k', ls='--')
	h0  = ax2.plot( d[:J].mean(axis=0), color=colors[0], lw=5 )[0]
	h1  = ax2.plot( d[J:].mean(axis=0), color=colors[1], lw=5 )[0]
	leg = ax2.legend( [h0,h1,h2], glabels + ['Null displacement'], loc='upper right', bbox_to_anchor=leg_loc[2]  )
	plt.setp( leg.get_texts(), name=fontname)


	# SPM results:
	ti.plot(  ax=ax3 )
	T2i.plot( ax=ax4 )
	tri.plot( ax=ax5 )
	twi.plot( ax=ax6 )
	

	# init axes decorations:
	for ax in AX:
		plt.setp( ax.get_xticklabels() + ax.get_yticklabels(), name=fontname, size=10 )
		ax.set_xlim(0, 100)
		ax.set_ylabel(None)
	[ax.set_xticklabels([])  for ax in [ax1,ax2,ax4]]
	
	
	
	# axis labels:
	sz = 16
	ax0.set_ylabel(dvlabel, name=fontname, size=sz)
	ax1.set_ylabel(dvlabel, name=fontname, size=sz)
	ax2.set_ylabel('Warp magnitude', name=fontname, size=sz)
	ax3.set_ylabel('SPM{t}', name=fontname, size=sz)
	ax4.set_ylabel(r'SPM{ $T^2$ }', name=fontname, size=sz)
	ax5.set_ylabel('SPM{t}', name=fontname, size=sz)
	[ax.set_xlabel(xlabel, name=fontname, size=sz)  for ax in [ax3,ax5,ax6]]


	# panel labels:
	labels = ['A.1', 'B.1', 'B.2', 'A.2', 'B.3', 'B.4', 'B.5']
	slabels = ['Linearly registered', 'Nonlinearly registered', 'Displacement fields', 'Statistical analysis', 'Main test  (amplitude + timing)', 'Post hoc  (amplitude)', 'Post hoc  (timing)']
	[ax.text(0.03, 0.92, f'({s})  {ss}', name=fontname, size=14, transform=ax.transAxes)   for ax,s,ss in zip( AX, labels, slabels ) ]
	tx0 = ax1.text(0.01, 1.05, 'Amplitude effects', ha='left', transform=ax1.transAxes)
	tx1 = ax2.text(0.99, 1.05, 'Timing effects', ha='right', transform=ax2.transAxes)
	plt.setp( [tx0,tx1] , name=fontname, size=16  )

	# background panels:
	c0,c1  = '0.6', '0.9'
	patch0 = patches.Rectangle([0.035,0.005], 0.328, 0.99, facecolor=c0, edgecolor=c0, alpha=0.9, zorder=-1)
	patch1 = patches.Rectangle([0.370,0.005], 0.628, 0.99, facecolor=c1, edgecolor=c1, alpha=0.9, zorder=-1)
	tx0    = fig.text(0.20, 0.97, '(A) Common univariate approach', ha='center')
	# tx1    = fig.text(0.20, 0.48, '( No explicit temporal\neffect consideration )', ha='center')
	tx1    = fig.text(0.20, 0.48, '( None )', ha='center')
	tx2    = fig.text(0.55, 0.97, '(B) Proposed multivariate approach')
	fig.add_artist(patch0)
	fig.add_artist(patch1)
	plt.setp( [tx0, tx1, tx2], name=fontname, size=20)
	
	x      = 0.01 
	y      = np.array(axy) + 0.5*axh
	tx0    = fig.text(x, y[0], 'Dependent variables')
	tx1    = fig.text(x, y[1], 'Multivariate analysis')
	tx2    = fig.text(x, y[2], 'Univariate analysis')
	plt.setp( [tx0, tx1, tx2], name=fontname, size=20, rotation=90, va='center')
	# tx1.set_size=14


	# axis limits:
	if ylim is not None:
		[ax.set_ylim(*yy)  for ax,yy in zip(AX, ylim)]

	
	def add_threshold_label(ax, x0, ti):
		s0,s1     = r'$\alpha$ < 0.05', r'$\alpha$ > 0.05'
		hax       = 0.02
		x,y0      = data2axes( ax, [x0, ti.zstar] )
		tx0       = ax.text(x, y0+hax, s0, va='bottom')
		tx1       = ax.text(x, y0-hax, s1, va='top')
		tx        = [tx0,tx1]
		plt.setp( tx, size=11, name=fontname, transform=ax.transAxes)
		return tx
		
		
	
	# add threshold labels:
	if alpha_x is not None:
		add_threshold_label( ax3, alpha_x[0], ti )
		add_threshold_label( ax4, alpha_x[1], T2i )
		add_threshold_label( ax5, alpha_x[2], tri )
		add_threshold_label( ax6, alpha_x[3], twi )
	
	
	

	return fig,AX
