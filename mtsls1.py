import numpy as np 

'''
Interface:
	x = np.random.uniform(low=info["lower"],high=info["upper"],size=info["dimension"])
	fx = fobj(x)
	ls1 = mtsls1(info,step=0.2)
	x,fx,FEs_computed = ls1.enhance(fobj,info,x,fx,FEs=25000)
'''
class mtsls1(object):
	def __init__(self,info,checkpoints,step=0.2):
		#vectorize lower and upper since they are scalar due to all variables have the same limit constraints 
		lowerbound = np.array([info["lower"]]*info["dimension"])
		upperbound = np.array([info["upper"]]*info["dimension"])
		#Set parameters 
		self.step = step
		self.improvement = np.zeros(info["dimension"])
		self.SR = np.array([(ub-lb)*step for lb,ub in zip(lowerbound,upperbound)])
		self.checkpoints = checkpoints.copy() 
		self.flag = False #it'll be set to true only if the last checkpoint is reached 

	def enhance(self,fobj,info,x,fx,results,control,FEs=25000):
		self.results = results.copy()
		self.control = control #needed for knowing when it's time for recording the best solution at hand when reaching some checkpoint
		counter = 0
		'''
		Warm-up
		'''
		x,fx,counter = self.warm_up(fobj,info,x,fx,counter)
		'''
		Search
		'''
		#Set a dimension list based on improvements made during warm-up. Arrangement is done from greater to lower improvement
		dim_list = [i for i in range(info["dimension"])]
		dim_sorted = np.array([k for _,k in sorted(zip(self.improvement,dim_list),reverse=True)])
		i = 0
		while counter < FEs and self.flag == False:
			dim = dim_sorted[i]
			y,fy,counter = self.improve_dim(fobj,info,x,fx,dim,counter)
			self.improvement[dim] = max(fx-fy,0.0)
			j = (i + 1) % info["dimension"]
			next_dim = dim_sorted[j]
			if self.improvement[dim] > 0.0:
				x = y.copy()
				fx = fy
				if self.improvement[dim] < self.improvement[next_dim]:
					dim_sorted = np.array([k for _,k in sorted(zip(self.improvement,dim_list),reverse=True)])
			else:
				self.SR[dim] /= 2.0
				i = j
				if self.SR[dim] < 1e-15:
					self.SR[dim] = (info["upper"]-info["lower"])*self.step
		#End
		return x,fx,counter,self.results

	def warm_up(self,fobj,info,x,fx,counter):
		dim_perm = np.random.permutation(info["dimension"])
		for dim in dim_perm:
			y,fy,counter = self.improve_dim(fobj,info,x,fx,dim,counter)
			self.improvement[dim] = max(fx-fy,0.0)
			if self.improvement[dim] > 0.0:
				x = y.copy()
				fx = fy
			else:
				self.SR[dim] /= 2.0
			if self.flag == True:
				break
		return x,fx,counter

	def improve_dim(self,f,info,x,fx,i,counter):
		'''
		x --> decision vecor 
		fx --> value of such decision vector  
		'''
		trial = x.copy()
		trial[i] -= self.SR[i]
		#Take care of limit constraints 
		if trial[i] > info["upper"]:
			trial[i] = (info["upper"]+x[i])/2
		elif trial[i] < info["lower"]:
			trial[i] = (info["lower"]+x[i])/2
		#Evaluate new candidate solution
		trial_value = f(trial)
		counter += 1
		#Save solution if status meets some checkpoint
		status = self.control + counter  
		if status in self.checkpoints:
			if fx <= trial_value:
				self.results.append((status,x.copy(),fx))#record best
			else:
				self.results.append((status,trial.copy(),trial_value))#record best
		#Set flag to true if last checkpoint is reached and terminate
		if status >= self.checkpoints[-1]: 
			self.flag = True
			if fx <= trial_value:
				return x,fx,counter
			else:
				return trial,trial_value,counter
		#Make a decision 
		if(trial_value < fx): #if new candidate better than old candidate, then...
			x = trial.copy()
			fx = trial_value
		elif(trial_value > fx): #if not better, then perform another trial
			trial = x.copy()
			trial[i] += 0.5*self.SR[i]
			#Take care of limit constraints
			if trial[i] > info["upper"]:
				trial[i] = (info["upper"]+x[i])/2
			elif trial[i] < info["lower"]:
				trial[i] = (info["lower"]+x[i])/2 
			#Evaluate new candidate solution
			trial_value = f(trial)
			counter += 1
			#Save solution if status meets some checkpoint
			status = self.control + counter  
			if status in self.checkpoints:
				if fx <= trial_value:
					self.results.append((status,x.copy(),fx))#record best
				else:
					self.results.append((status,trial.copy(),trial_value))#record best
			#Save new candidate if better than the old candidate 
			if(trial_value < fx):
				x = trial.copy()
				fx = trial_value
		#End 
		if status >= self.checkpoints[-1]: #set flag to true if last checkpoint is reached
			self.flag = True
		return x,fx,counter
