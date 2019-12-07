# Installation

This software uses CUDA Toolkit 10 and contains a Visual Studio project in the `CoDi` folder. An NVIDIA GPU like the ones used in the labs is required to run this software.

This is available in a [GitHub repository.](https://github.com/JRegimbal/ECSE-420-Project)

# Running

The software takes one command line parmeter, which is the dimension of the grid. For example to use a 100 by 100 by 100 chromosome, the command line argument should be 100. The chromosome is included in the `chromosome.h` header file.

The number of threads is set by the global `threads` variable in `kernel.cu`. Since this actually controls the number of threads per block, the maximum supported value is 1024. Nothing else is required to run the growth phase of CoDi.

The neuron density can be changed by setting the `neuronSeedP` variable. It is currently set to 10%.

# Files

* `kernel.cu`: The main file containing `main()` and the kernel used for the growth phase.
* `chromosome.h`: The chromosome loaded for CoDi expressed as an array of unsigned characters. These values are converted to instruction types.
* `codi_cell.h`: Enumerations and the Cell class itself. These are what make up the generated grid and become the cells in the neural network. Functions marked with `__device__` can only be run on the GPU.
* `chromosome_10x10x10.h`: The 10x10x10 chromosome used during earlier tests.
