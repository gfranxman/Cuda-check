# i used conda for the environment
# $ conda create -p ./cudaenv
# $ conda init
# $ conda activate ./cudaenv
# $ conda install numba
# $ conda install cudatoolkit
# $ python p.py
# /home/ubuntu/app/t/p.py:12: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
#  @jit(target_backend='cuda')
#  without GPU: 2.8261743299663067
#  with GPU: 0.2290111150359735

# in gh workspace:
# (/workspaces/Cuda-check/cudaenv) @gfranxman ➜ /workspaces/Cuda-check (main) $ python ./cuda-check.py 
# /workspaces/Cuda-check/./cuda-check.py:25: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
#   @jit(target_backend='cuda')
# without GPU: 2.0189463680000017
# with GPU: 0.7738111160000472



from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer 

# normal function to run on cpu
def func(a):                                                     
        for i in range(10000000):
                a[i]+= 1        

# function optimized to run on gpu 
@jit(target_backend='cuda')                                              
def func2(a):
        for i in range(10000000):
                a[i]+= 1
if __name__=="__main__":
        n = 10000000                                            
        a = np.ones(n, dtype = np.float64)
        
        start = timer()
        func(a)
        print("without GPU:", timer()-start) 
        
        start = timer()
        func2(a)
        print("with GPU:", timer()-start)

