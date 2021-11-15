
import os,pathlib
import numpy as np
from matplotlib import pyplot as plt
import spm1d
import nlreg1d


dirDATA   = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Data')
niter     = 5
save      = True



mlabels   = ['Semimem', 'Semiten', 'BicepsFemorisLH', 'BicepsFemorisSH', 'RectusFemoris', 'Vasmed', 'Vasint', 'Vaslat', 'medGastroc', 'latGastroc']
ylimdata  = [(-0.05, 1.5), (-0.05, 0.4), (-0.05, 0.9), (-0.05, 0.4), (-0.05, 0.4),   (-0.05, 1.2), (-0.05, 1.2), (-0.05, 1.5), (-0.05, 1.9), (-0.05, 1.5)]
for i,mlabel in enumerate(mlabels):
	dataset   = spm1d.data.mv1d.hotellings2.Besier2009muscleforces()
	y0,y1     = dataset.get_data()  #A:Controls, B:PFP
	y0,y1     = 0.001*y0[:,:,i], 0.001*y1[:,:,i]
	y         = np.vstack( [y0, y1] )
	J         = y0.shape[0]
	### register:
	Q         = y.shape[1]
	q         = np.linspace(0, 1, Q)
	yr,wr     = nlreg1d.reg.fpca( y, ncomp=5, smooth=False, niter=niter )
	### plot:
	plt.close('all')
	colors    = '0.0', (0.8,0.1,0.1)
	xlabel    = 'Time  (% stance)'
	ylimd     = ylimdata[i]
	ylimt     = (-6, 6)
	ylim      = [ ylimd, ylimd, (-50, 70), ylimt, (-2, 45), ylimt, ylimt ]
	alpha_x   = [70, 70, 70, 70]
	leg_loc   = [(0.80, 0.92), (0.80, 0.92), (0.99, 0.99)]
	fig,AX    = nlreg1d.plot.plot_multipanel(y, yr, wr, J, colors, ylim, alpha_x, paired=False, dvlabel='Muscle force  (kN)', group_labels=['Control mean', 'PFP mean'], xlabel=xlabel, leg_loc=leg_loc)
	fig.text(0.37, 0.5, mlabel, bbox=dict(facecolor='0.9'), size=24, ha='center', fontname='Helvetica')
	plt.show()
	if save:
		dirFIGS  = os.path.join( pathlib.Path( __file__ ).parent.parent, 'Figures')
		plt.savefig( os.path.join(dirFIGS, f'supp_datasetC_{i}_{mlabel}.pdf')  )


