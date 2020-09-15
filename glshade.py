#encoding: utf-8 
from cec2013lsgo.cec2013 import Benchmark #documentation and package installation steps: https://pypi.org/project/cec2013lsgo/
import numpy as np 
from mtsls1 import mtsls1
from shade import shade
from eshadels import eshadels

if __name__ == "__main__":

	#Control parameters
	number_executions = 1 #for multiples executions per function: set number_executions>1 !!!!!
	Functions = [12] #np.array([f for f in range(15)]) + 1 #use the previous commented line: if you want to test all functions!!!!!
	seeds = (np.linspace(0,1,num=number_executions)*100).astype(int) #generate a set of seed according with the number of executions set

	#GL-SHADE main parameters
	popsize1 = 100
	popsize2 = 100
	maxFEs = 3e6
	checkpoints = (np.array([0.04,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])*maxFEs).astype(int) #array

	for fid in Functions:
		#print("Processing f" + str(fid))
		filename = "myresults_" + str(popsize1) + "pop1_" + str(popsize2) + "pop2_f" + str(fid) + ".csv"
		outputfile = open(filename,"w+")
		outputfile.write("FEs,Value,Seed,Function\n")
		for Rseed in seeds:
			'''
			Set objective function: the CEC'13 LSGO Benchmark presents 15 minimization problems.  
			''' 
			bench = Benchmark()
			fobj = bench.get_function(fid) #objective function
			info = bench.get_info(fid) #info is a dictionary which keys are lower, upper, dimension, threshold and best.

			'''
			Set parameters
			'''
			np.random.seed(Rseed)
			currentFEs = 0
			results = [] #list of tuples. Each tuple will be of the form (checkpoint,decision_vector,value_of_decision_vector) 

			'''
			Set components
			'''
			DE1 = shade(fobj,info,popsize1,param_memsize=popsize1,checkpoints=checkpoints[:])
			DE2 = eshadels(fobj,info,popsize2,param_memsize=popsize2,checkpoints=checkpoints[:])
			LS = mtsls1(info,checkpoints[:],step=0.2)
			computedFEs = popsize1 + popsize2
			currentFEs += computedFEs

			'''
			Initialize best solution so far
			'''
			if DE1.bestvalue <= DE2.bestvalue:
				xbest = DE1.bestind.copy()
				fxbest = DE1.bestvalue
			else:
				xbest = DE2.bestind.copy()
				fxbest = DE2.bestvalue

			'''
			Optimization 
			'''
			#Enhacement via early local search 
			xbest,fxbest,computedFEs,results = LS.enhance(fobj,info,xbest,fxbest,results,currentFEs,FEs=25000)
			currentFEs += computedFEs #update number of function evaluations computed 

			#Evolution stage 
			while currentFEs < checkpoints[-1]:
				xbest,fxbest,computedFEs,results = DE1.evolve(fobj,info,xbest,fxbest,results,currentFEs,FEs=25000)
				currentFEs += computedFEs
				xbest,fxbest,computedFEs,results = DE2.evolve(fobj,info,xbest,fxbest,results,currentFEs,FEs=25000)
				currentFEs += computedFEs

			''' 
			Save results
			'''
			for checkpoint,sol,value in results:
				outputfile.write("%d,%.6e,%d,f%d\n" % (checkpoint,value,Rseed,fid))
		outputfile.close()

