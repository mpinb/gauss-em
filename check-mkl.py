
import numpy as np
import os

# this might not indicate anything about the mkl.
np.show_config()

# this should print mkl libraries loaded (after numpy imported).
os.system('cat /proc/'+str(os.getpid()).strip()+'/maps|grep libmkl|awk '+"'{print $NF}'"+'|uniq')
