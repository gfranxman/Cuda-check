# Cuda-check

Scripts I'm using to check we're set up to use the gpu.

We're using conda here.

## setup
    $ conda create -p ./cudaenv
    $ conda init
    $ conda activate ./cudaenv
    $ conda install numba
    $ conda install cudatoolkit

## run
    $ conda activate ./cudaenv
    $ python ./cuda-check.py

It should be around a 10x improvement with the GPU