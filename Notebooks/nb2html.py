
'''
Export notebooks in HTML format
'''


import os,glob
import nbformat,nbconvert


def export_nb(fnameNB, fnameHTML):
	nb        = nbformat.read(fnameNB, as_version=4)
	exporter  = nbconvert.HTMLExporter()
	(bod,res) = exporter.from_notebook_node(nb)
	with open(fnameHTML, 'w', encoding='utf8') as fid:
		fid.write(bod)


dirNB        = os.path.dirname( __file__ )
dirHTML      = os.path.join( dirNB, 'html' )
names        = os.listdir( dirNB )
for name in names:
	if name.endswith('.ipynb'):
		fpathNB   = os.path.join( dirNB, name )
		fpathHTML = os.path.join( dirHTML, os.path.splitext(name)[0] + '.html' )
		print( f'Exporting {name}...' )
		export_nb( fpathNB, fpathHTML )
print('Done.\n\n')
