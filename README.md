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

It should be around a 10x improvement with the GPU.






## for torch test I used
    $ conda create -p ./cudaenv
    $ conda init
    $ conda activate ./cudaenv
    $ conda install numba
    $ conda install cudatoolkit

    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    $ conda install --file requirements-gpu.in 
    $ pip  install weaviate-client
    $ python ~/app/t/p.py    # verify cuda and numba
    $ python ./torch-check.py  # verify torch will use cuda
    $ chainlit run app.py 
