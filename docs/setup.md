## Setting up the system

We assume cuda drivers and [cuDNN](https://developer.nvidia.com/cudnn) is installed. cuda installation instructions are [here](https://askubuntu.com/questions/799184/how-can-i-install-cuda-on-ubuntu-16-04). cuDNN installation instructions are [described below](#cudnn-instllation).  

### Conda

For maintaining a virtual environment in python, we use `conda`. Conda can be installed by [downloading this file](https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh) and using `bash <installer.sh>`. Download the latest version from [this link](https://www.continuum.io/downloads).

### Creating a Virtual Environment using Conda

- `conda create --name dl python=3 <keras tensorflow pytorch>`

- While this will install the required libraries, you should install tensorflow from source. The instructions are [here](#installing-tensorflow-from-source). 

- **TODO**: Document `pytorch` installation from source.
 
- `source activate dl` to activate the virtualenv.

### Installing Tensorflow from Source

- **Clone tensorflow**: `git clone https://github.com/tensorflow/tensorflow `

- **Install Bazel**: [See the instructions](https://docs.bazel.build/versions/master/install.html)
- **Configure**: `cd tensorflow && ./configure`. During the configuration, you would be asked many questions, just use carriage return, other than the cuda question. 
```
Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
"Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 7.0.1
Please specify the location where cuDNN 7.0.1 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 5.2]
Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 
Do you wish to build TensorFlow with MPI support? [y/N]: 
No MPI support will be enabled for TensorFlow.

``` 
_Please specify a list of comma-separated Cuda compute capabilities you want to build with_ : Look [here](https://developer.nvidia.com/cuda-gpus) to get the compute capability value. 
 
- **Build**: 
```buildoutcfg
bazel build --config opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

- **Install the python binding**: `pip install -I /tmp/tensorflow_pkg/tensorflow-*.whl 
`  

### cuDNN Installation

- cuDNN installers can be downloaded from [this link](https://developer.nvidia.com/rdp/cudnn-download) but you would need to create a developer account for that.

- Once you can access the proper page, download the following files:

```buildoutcfg
cuDNN v7.0 Runtime Library for Ubuntu<14/16>.04 (Deb)

cuDNN v7.0 Developer Library for Ubuntu<14/16>.04 (Deb)

cuDNN v7.0 Code Samples and User Guide for Ubuntu<14/16>.04 (Deb)
```
_download the ones for your ubuntu version_. Use `sudo dpkg -i` to install. to check the installation, follow these steps:

```buildoutcfg
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```
It should show `Test Passed`.

- cuDNN `so` files are typically at `/usr/lib/x86_64-linux-gnu/`. Copy them to your cuda installation, typically at `/usr/local/cuda/lib64`. Note the version (eg. _7.0.1_) because that would be needed to build tensorflow. 

- Copy the header file to the proper location `cp /usr/include/x86_64-linux-gnu/cudnn_v7.h /usr/local/cuda/include/cudnn.h`. 

- Also, add the following lines to `~/.bashrc` if not there:
```buildoutcfg
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:{LD_LIBRARY_PATH}

```




 