import numpy as np 

def correct_Fi(Fi,muFi):
	while Fi <= 0.0:
		Fi = np.random.normal(muFi,0.1)
	return Fi

def correct_Cr(Cr):
	Cr = np.where(Cr>1.0,1.0,Cr)
	Cr = np.where(Cr<0.0,0.0,Cr)
	return Cr

def correct_dim(info,z,x):
	z = np.where(z>info["upper"],(info["upper"]+x)/2,z)
	z = np.where(z<info["lower"],(info["lower"]+x)/2,z)
	return z

def correct_archive(extpop,popsize):
	extpopsize = len(extpop)
	while extpopsize > popsize:
		rnd = np.random.randint(extpopsize)
		del extpop[rnd]
		extpopsize -= 1
	return extpop

def WAmean(Cr,delta):
	#Cr and delta must be numpy arrays 
	sumdelta = sum(delta)
	result = sum((delta/sumdelta)*Cr)
	if result > 1:
		result = 1
	elif result < 0:
		result = 0
	return result

def WLmean(F,delta):
	#F and delta must be numpy arrays
	sumdelta = sum(delta)
	tmp1 = sum((delta/sumdelta)*(F*F))
	tmp2 = sum((delta/sumdelta)*F)
	result = tmp1/tmp2
	if result > 1:
		result = 1
	elif result < 0:
		result = 0
	return result

def generate_adaptive_parameters(Fmem,Crmem,popsize):
	#Choose mu (mean) randomly from memory  
	muF = np.random.choice(Fmem,size=popsize) 
	muCr = np.random.choice(Crmem,size=popsize)
	#Generate parameter and make sure it meets the limit constraint 
	F = np.random.normal(muF,0.1) #F is an array 
	F = np.array([correct_Fi(Fi,muFi) for Fi,muFi in zip(F,muF)])
	Cr = correct_Cr(np.random.normal(muCr,0.1)) #Cr is an array
	return F,Cr

def generate_random_data(popsize,intUextpopsize,pmin,pmax,dimension):
	#a and b are randomly chosen indices from internal population range and {internal UNION external} population range, respectively
	a = np.random.randint(popsize,size=popsize) #needed during mutation 
	b = np.random.randint(intUextpopsize,size=popsize) #needed during mutation
	#generate pbest 
	pbest_selection_range = (popsize*np.random.uniform(low=pmin,high=pmax,size=popsize)).astype(int) #needed for generating pbest
	pbest = np.random.randint(pbest_selection_range) #needed during mutation
	#generate jrand
	jrand = np.random.randint(dimension,size=popsize) #needed during recombination 
	return a,b,pbest,jrand


class shade(object):
	def __init__(self,fobj,info,popsize,param_memsize,checkpoints):
		self.popsize = popsize
		self.param_memsize = param_memsize
		self.param_memindex = 0
		self.pmin = 2.0/popsize
		self.pmax = 0.2
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

	def receive(self,xbest,fxbest):
		#Replace old best with the incoming new best
		self.pop[self.bestindex] = xbest.copy()
		self.values[self.bestindex] = fxbest 
		#Update best solution info
		self.bestind = self.pop[self.bestindex].copy()
		self.bestvalue = self.values[self.bestindex]

	def evolve(self,fobj,info,xbest,fxbest,results,control,FEs=25000):
		'''
		SHADE's algoritm 
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
				v = x + F[i]*(self.pop[idx_sorted[pbest[i]]] - x) + F[i]*(self.pop[a[i]] - intUext[b[i]])
				v = correct_dim(info,v,x) #correct jth variable if needed (make sure meets the limit constraint)
				#Recombination (binomial)
				u = np.where(np.random.uniform(size=info["dimension"])<=Cr[i],v,x)
				u[jrand[i]] = v[jrand[i]]
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
		#End
		return self.bestind,self.bestvalue,counter,self.results
