
import os,pathlib

dirREPO = str( pathlib.Path( __file__ ).parent.parent )
dirDATA = os.path.join( dirREPO, 'Data' )
dirFIGS = os.path.join( dirREPO, 'Figures' )


from . import data
from . import reg
from . import warp
from . import plot
from . import util

Warp1D          = warp.Warp1D
Warp1DList      = warp.Warp1DList
random_warp     = warp.random_warp
register_linear = reg.linear
register_srsf   = reg.srsf



