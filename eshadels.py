import numpy as np 
from shade import correct_Fi,correct_Cr,correct_dim,correct_archive,WAmean,WLmean,generate_adaptive_parameters,generate_random_data

class eshadels(object):
	def __init__(self,fobj,info,popsize,param_memsize,checkpoints):
		self.popsize = popsize
		self.param_memsize = param_memsize
		self.param_memindex = 0
		self.pmin = 2.0/popsize
		self.pmax = 0.1
		self.pop = [np.random.uniform(low=info["lower"],high=info["upper"],size=info["dimension"]) for i in range(popsize)]
		self.values = np.array([fobj(ind) for ind in self.pop])
		self.Fmem = np.array([0.5]*param_memsize)
		self.Crmem = self.Fmem.copy()
		self.bestindex = np.argmin(self.values)
		self.bestind = self.pop[self.bestindex].copy()
		self.bestvalue = self.values[self.bestindex]
		self.extpop = []
		self.idx = [i for i in range(self.popsize)]
		self.checkpoints = checkpoints.copy()
		self.wmin = 0.0
		self.wmax = 0.2

	def receive(self,xbest,fxbest):
		#Place incoming new best at a position chosen randomly 
		self.bestindex = np.random.randint(self.popsize)
		self.pop[self.bestindex] = xbest.copy()
		self.values[self.bestindex] = fxbest 
		#Update best solution info
		self.bestind = self.pop[self.bestindex].copy()
		self.bestvalue = self.values[self.bestindex]

	def edels(self,fobj,info,counter):
		#FEs = control + counter 
		if (self.control+counter)<self.checkpoints[-1]:
			#Prepare random data
			k = np.random.randint(self.popsize,size=info["dimension"])
			n = np.random.randint(info["dimension"],size=info["dimension"])
			rnd1 = np.random.uniform(size=info["dimension"])
			rnd2 = np.random.uniform(size=info["dimension"])
			#Enhance best solution so far
			for j in range(info["dimension"]):
				#Force diversity
				while k[j] == self.bestindex:
					k[j] = np.random.randint(self.popsize)
				while n[j] == j:
					n[j] = np.random.randint(info["dimension"])
				#local search stage
				mu = self.bestind.copy()
				r2 = self.wmin + ((self.control+counter)/self.checkpoints[-1])*(self.wmax-self.wmin)
				if rnd1[j] <= r2:
					mu[j] = self.bestind[n[j]] + (2*rnd2[j]-1)*(self.bestind[n[j]] - self.pop[k[j]][n[j]])
				else:
					mu[j] = self.bestind[j] + (2*rnd2[j]-1)*(self.bestind[n[j]] - self.pop[k[j]][n[j]])
				mu = correct_dim(info,mu,self.bestind)
				#Evaluate new candidate 
				mu_value = fobj(mu)
				counter += 1 
				#Take the best between mu and bestind as the new bestind
				if mu_value <= self.bestvalue:
					#bestindex remains the same 
					self.bestind = mu.copy()
					self.bestvalue = mu_value
					self.pop[self.bestindex] = self.bestind.copy()
					self.values[self.bestindex] = self.bestvalue 
				#Save best solution if status meets some checkpoint
				status = self.control + counter  
				if status in self.checkpoints:
					self.results.append((status,self.bestind.copy(),self.bestvalue))#record it
				#Terminate immediately if last checkpoint is reached 
				if status >= self.checkpoints[-1]: 
					return counter 
		return counter


	def evolve(self,fobj,info,xbest,fxbest,results,control,FEs=25000):
		'''
		eSHADE_ls's algoritm 
		'''
		self.receive(xbest,fxbest) #integrate incoming best solution to population
		self.results = results.copy()
		self.control = control 
		counter = 0
		#Search 
		while counter<FEs and (self.control+counter)<self.checkpoints[-1]:
			#Reset 
			next_pop = []
			next_values = []
			storage_F = []
			storage_Cr = []
			delta = []
			#Joining internal and external population 
			intUext = self.pop + self.extpop
			#Create a ranking list
			idx_sorted = [i for _,i in sorted(zip(self.values,self.idx),reverse=False)] #reverse is False since i'm minimizing 
			#Get F and CR 
			F,Cr = generate_adaptive_parameters(self.Fmem,self.Crmem,self.popsize)
			#Random data needed during mutation and recombination   
			a,b,pbest,jrand = generate_random_data(self.popsize,len(intUext),self.pmin,self.pmax,info["dimension"])
			#For every individal in population...
			for i,x,fx in zip(self.idx,self.pop,self.values):
				#Force diversity 
				while a[i] == i:
					a[i] = np.random.randint(self.popsize)
				while b[i] == i or b[i] == a[i]:
					b[i] = np.random.randint(len(intUext))
				#Mutation
				v = self.pop[idx_sorted[pbest[i]]] + F[i]*(self.pop[a[i]] - intUext[b[i]])
				v = correct_dim(info,v,x) #correct jth variable if needed (make sure meets the limit constraint)
				#Recombination (exponential) [slow implementation, it can be enhanced]
				u = x.copy()
				j = jrand[i]
				u[j] = v[j]
				j = (j+1) % info["dimension"]
				L = 1
				rnd = np.random.uniform(size=info["dimension"])
				while(rnd[j]<=Cr[i] and L<info["dimension"]):
					u[j] = v[j]
					j = (j+1) % info["dimension"]
					L += 1
				#Selection 
				fu = fobj(u)
				counter += 1
				if fu <= fx: #if better than target vector 
					if fu < fx: #if strictly better, then:
						self.extpop.append(x.copy()) #add defeated target vector to the external archive
						storage_F.append(F[i]) #record F
						storage_Cr.append(Cr[i]) #record Cr
						delta.append(fx - fu) #record difference/improvement 
					#advance trial vector to next generation 
					next_pop.append(u.copy())
					next_values.append(fu)
				else:
					next_pop.append(x.copy())
					next_values.append(fx)
				#Save best solution if status meets some checkpoint
				status = self.control + counter  
				if status in self.checkpoints:
					bidn = np.argmin(next_values) #best index next population 
					if next_values[bidn] <= self.bestvalue: #if the best in next population better than the best in current population
						self.results.append((status,next_pop[bidn].copy(),next_values[bidn]))#record it
					else:
						self.results.append((status,self.bestind.copy(),self.bestvalue))#record it
				#Terminate immediately if last checkpoint is reached 
				if status >= self.checkpoints[-1]: 
					bidn = np.argmin(next_values) #best index next population 
					if next_values[bidn] <= self.bestvalue: #if the best in next population better than the best in current population
						return next_pop[bidn],next_values[bidn],counter,self.results
					else:
						return self.bestind,self.bestvalue,counter,self.results 
			#advance generation 
			self.pop = next_pop.copy()
			self.values = next_values.copy()
			#Update best solution info 
			self.bestindex = np.argmin(self.values)
			self.bestind = self.pop[self.bestindex].copy()
			self.bestvalue = self.values[self.bestindex]
			#Correct external archive if needed 
			self.extpop = correct_archive(self.extpop,self.popsize)
			#Processing the history of successful parameters
			if len(storage_F) > 0 and len(storage_Cr) > 0: #if they are not empty 
				self.Fmem[self.param_memindex] = WLmean(np.array(storage_F),np.array(delta))
				self.Crmem[self.param_memindex] = WAmean(np.array(storage_Cr),np.array(delta))
				self.param_memindex = (self.param_memindex + 1) % self.param_memsize
			#Apply local search according to EDE-LS's algorithm
			counter = self.edels(fobj,info,counter)
		#End
		return self.bestind,self.bestvalue,counter,self.results
