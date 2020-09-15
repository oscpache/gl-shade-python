# gl-shade-python
This source code implements the GL-SHADE algorithm presented in the paper "A SHADE-Based Algorithm for Large Scale Global Optimization" at the PPSN 2020 international conference. Here we present a non parallel implementation of GL-SHADE using python.

### gl-shade-cuda
Public interested in a parallel implementation of GL-SHADE using CUDA, you can visit https://github.com/delmoral313/gl-shade. 

## Getting Started

### Prerequisites
In order to run the code, you need python 3 installed on your computer and the following python packages: 
- numpy 
- cec2013lsgo  

Documentation and steps for installing the cec2013lsgo package can be consulted here (https://pypi.org/project/cec2013lsgo/).

## Running the program 
For running the program you just need to type the following command on terminal:
- $ python3 glshade.py 

### Selecting the objective function and number of executions
By default the program drives one execution per objective function and set the 12th problem of the CEC'13 LSGO Benchmark as the objective function. These parameters can be changed easily. For doing so take a look at lines 11 and 12 of the main program (glshade.py) and realize that there are two control parameters which can be modified according our needs, named number_executions and Functions. number_executions must be an integer greater than or equal to 1 while Functions must be a list or an array of integers. Here, number_executions controls how many runs per objective function are made (different rng seeds are used for different runs) and Functions stores all the problems that'll be adopted as objective function (keep in mind that the CEC'13 LSGO Benchmark includes only 15 problems).   

## Output files
When the glshade.py run is over two output files are generated per objective function, named myresults_100pop1_100pop2_f[objfunc].csv and results_f[objfunc].csv  
