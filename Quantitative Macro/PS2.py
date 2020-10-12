import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import LinearConstraint,Bounds,minimize

#Defining a class for the first model
class model1:
	
	def __init__(self, beta, theta, h, delta, y_ss):
		self.beta,self.theta,self.h,self.delta,self.y_ss=beta,theta,h,delta,y_ss
		
	def u(self,c):
		"Evaluating utility function"
		return np.log(c)
	
	def y(self,k,z):
		"evaluating production function"
		return k**(1-self.theta)*(z*self.h)**(self.theta)
	def obtain_z(self):
		"obtaining z in the steady states"
		k=4*self.y_ss
		z=self.y_ss**(1/self.theta)*k**((self.theta-1)/self.theta)/self.h
		self.z1=z
		self.z2=z*2
		return [z,z*2]
	
	
	def steady_state(self,new_z):
		"computing steady state given z"
		z=new_z
		k=new_z*self.h*(((1/self.beta)-1+self.delta)/(1-self.theta))**(-1/self.theta)
		y=k**(1-self.theta)*(new_z*self.h)**(self.theta)
		i=self.delta*k
		c=y-i
		beta=self.beta
		theta=self.theta
		h=self.h
		delta=self.delta
		list_ret=[]
		variables=[z,y,k,i,c,beta, theta, h, delta]
		for var in variables:
			list_ret.append(var)
		return pd.DataFrame(list_ret,index=["z","y","k","i","c","beta", "theta", "h", "delta"],columns=["Steady state"])
	
	def transition(self,init_y,init_k,init_i,new_z,time=50):
		"Transition path for 50 periods"		
		k=np.empty(time+1)
		y=np.empty(time+1)
		i=np.empty(time+1)
		c=np.empty(time+1)
		k[0]=init_k
		y[0]=init_y
		i[0]=init_i
		c[0]=init_y-init_i
		for t in range(time):
			k[t+1]=i[t]+(1-self.delta)*k[t]
			c[t+1]=c[t]*self.beta*((1-self.theta)*((new_z*self.h/k[t+1])**self.theta)+1-self.delta)
			y[t+1]=((new_z*self.h/k[t+1])**self.theta)*k[t+1]
			i[t+1]=y[t+1]-c[t+1]
		return y,k,i,c
	
	def transition_c(self,init_y,init_k,init_i,new_z,c_1,time=50):
		"Transition path for 50 periods with 'adjusted' c"		
		k=np.empty(time+1)
		y=np.empty(time+1)
		i=np.empty(time+1)
		c=np.empty(time+1)
		k[0]=init_k
		y[0]=init_y
		i[0]=init_i
		c[0]=init_y-init_i
		for t in range(time):
			if t==0:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[1]=c_1
				y[t+1]=((new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
			else:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[t+1]=c[t]*self.beta*((1-self.theta)*((new_z*self.h/k[t+1])**self.theta)+1-self.delta)
				y[t+1]=((new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
		return y,k,i,c
		
	def double_transition(self,init_y,init_k,init_i,new_z,new_new_z,c_1,c_10,time=47,t_shock=10):
		"Transition path for two shocks with adjusted c"		
		k=np.empty(time+1)
		y=np.empty(time+1)
		i=np.empty(time+1)
		c=np.empty(time+1)
		k[0]=init_k
		y[0]=init_y
		i[0]=init_i
		c[0]=init_y-init_i
		for t in range(time):
			if t==0:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[t+1]=c_1
				y[t+1]=((new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
			elif t<t_shock:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[t+1]=c[t]*self.beta*((1-self.theta)*((new_z*self.h/k[t+1])**self.theta)+1-self.delta)
				y[t+1]=((new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
			elif t==t_shock:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[t+1]=c_10
				y[t+1]=((new_new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
			else:
				k[t+1]=i[t]+(1-self.delta)*k[t]
				c[t+1]=c[t]*self.beta*((1-self.theta)*((new_new_z*self.h/k[t+1])**self.theta)+1-self.delta)
				y[t+1]=((new_new_z*self.h/k[t+1])**self.theta)*k[t+1]
				i[t+1]=y[t+1]-c[t+1]
		return y,k,i,c
		


#creating the first model		
m1=model1(1/1.02,0.67,0.31,0.0625,1)
#obtaining z's
m1.obtain_z()
#computing steady states
m1.steady_state(m1.z1)
m1.steady_state(m1.z2)
pd.concat([m1.steady_state(m1.z1),m1.steady_state(m1.z2)],axis=1)
#first transition path - wrong onr
transition_bad=m1.transition(m1.steady_state(m1.z1).iloc[1,0],
			  m1.steady_state(m1.z1).iloc[2,0],
			  m1.steady_state(m1.z1).iloc[3,0],
			  m1.steady_state(m1.z2).iloc[0,0])

fig, axs = plt.subplots(2, 2, figsize=(21, 8))
colors = ['blue', 'blue','blue','blue']
titles = ['Production', 'Capital','Investment','Consumption']
ylabels = ['$y_t$', '$k_t$','$i_t$','$c_t$']
T = transition_bad[0].size - 1
for i in range(2):
	for j in range(2):
		axs[i,j].plot(transition_bad[j+i*2], c=colors[j+i*2])
		axs[i,j].set(xlabel='t', ylabel=ylabels[j+i*2], title=titles[j+i*2])
		axs[i,j].axhline(m1.steady_state(m1.z2).iloc[1+j+2*i,0],color='k',ls='--',lw=1)
plt.show()

#defining object function to obtain the c
#it calculates the square of difference between variable value in the period 50 and steady state and then divide bu the steady state
def ob_func(c):
	path=m1.transition_c(m1.steady_state(m1.z1).iloc[1,0],
			  m1.steady_state(m1.z1).iloc[2,0],
			  m1.steady_state(m1.z1).iloc[3,0],
			  m1.steady_state(m1.z2).iloc[0,0],
			  c)
	SS1=((path[0][-1]-m1.steady_state(m1.z2).iloc[1,0])**2)/m1.steady_state(m1.z2).iloc[1,0]
	SS2=((path[1][-1]-m1.steady_state(m1.z2).iloc[2,0])**2)/m1.steady_state(m1.z2).iloc[2,0]
	SS3=((path[2][-1]-m1.steady_state(m1.z2).iloc[3,0])**2)/m1.steady_state(m1.z2).iloc[3,0]
	SS4=((path[3][-1]-m1.steady_state(m1.z2).iloc[4,0])**2)/m1.steady_state(m1.z2).iloc[4,0]
	return SS1+SS2+SS3+SS4

c_1=fsolve(ob_func,0)

#Obtaining the good transition path
transition_better=m1.transition_c(m1.steady_state(m1.z1).iloc[1,0],
			  m1.steady_state(m1.z1).iloc[2,0],
			  m1.steady_state(m1.z1).iloc[3,0],
			  m1.steady_state(m1.z2).iloc[0,0],
			  c_1)

fig, axs = plt.subplots(2, 2, figsize=(21, 8))
colors = ['blue', 'blue','blue','blue']
titles = ['Production', 'Capital','Investment','Consumption']
ylabels = ['$y_t$', '$k_t$','$i_t$','$c_t$']
T = transition_better[0].size - 1
for i in range(2):
	for j in range(2):
		axs[i,j].plot(transition_better[j+i*2], c=colors[j+i*2])
		axs[i,j].set(xlabel='t', ylabel=ylabels[j+i*2], title=titles[j+i*2])
		axs[i,j].axhline(m1.steady_state(m1.z2).iloc[1+j+2*i,0],color='k',ls='--',lw=1)
plt.show()

#Following similar approach, here we obtain a c for the second shock
def ob_func2(c):
	path=m1.double_transition(m1.steady_state(m1.z1).iloc[1,0],
			  m1.steady_state(m1.z1).iloc[2,0],
			  m1.steady_state(m1.z1).iloc[3,0],
			  m1.steady_state(m1.z2).iloc[0,0],
			  m1.steady_state(m1.z1).iloc[0,0],
			  c_1,c)
	SS1=((path[0][-1]-m1.steady_state(m1.z1).iloc[1,0])**2)/m1.steady_state(m1.z1).iloc[1,0]
	SS2=((path[1][-1]-m1.steady_state(m1.z1).iloc[2,0])**2)/m1.steady_state(m1.z1).iloc[2,0]
	SS3=((path[2][-1]-m1.steady_state(m1.z1).iloc[3,0])**2)/m1.steady_state(m1.z1).iloc[3,0]
	SS4=((path[3][-1]-m1.steady_state(m1.z1).iloc[4,0])**2)/m1.steady_state(m1.z1).iloc[4,0]
	return SS1+SS2+SS3+SS4

c_2=fsolve(ob_func2,0)

#Obtaining the transition path
transition_d=m1.double_transition(m1.steady_state(m1.z1).iloc[1,0],
			  m1.steady_state(m1.z1).iloc[2,0],
			  m1.steady_state(m1.z1).iloc[3,0],
			  m1.steady_state(m1.z2).iloc[0,0],
			  m1.steady_state(m1.z1).iloc[0,0],
			  c_1,c_2)

fig, axs = plt.subplots(2, 2, figsize=(21, 8))
colors = ['blue', 'blue','blue','blue']
titles = ['Production', 'Capital','Investment','Consumption']
ylabels = ['$y_t$', '$k_t$','$i_t$','$c_t$']
T = transition_d[0].size - 1
for i in range(2):
	for j in range(2):
		axs[i,j].plot(transition_d[j+i*2], c=colors[j+i*2])
		axs[i,j].set(xlabel='t', ylabel=ylabels[j+i*2], title=titles[j+i*2])
		axs[i,j].axhline(m1.steady_state(m1.z1).iloc[1+j+2*i,0],color='k',ls='--',lw=1)
plt.show()


#################################################################################
########### EXERCISE 2 ##########################################################
#################################################################################

# Defining a class for a covid model with a given parameters values
class model2:
	def __init__(self,Af=1,Anf=1,rho=1.1,kf=0.2,knf=0.2,omega=20,gamma=0.9,io=0.2,N=1):
		self.Af,self.Anf,self.rho,self.kf,self.knf,self.omega,self.gamma,self.io,self.N=Af,Anf,rho,kf,knf,omega,gamma,io,N

mtest=model2()

#defing an objective function to minimze , equation 11 from the pdf
def ob_func32(arguments):
	Hf=arguments[0]
	Hnf=arguments[1]
	Y=(mtest.Af*Hf**((mtest.rho-1)/mtest.rho)+ctw*mtest.Anf*Hnf**((mtest.rho-1)/mtest.rho))**(mtest.rho/(mtest.rho-1))
	return -(Y-mtest.kf*Hf-mtest.knf*Hnf-mtest.omega*(1-mtest.gamma)/mtest.N*beta*mtest.io*Hf**2)

# defining a constraints: Hf+Hnf<=1, Hf,Hnf belong  to [0,1] interval}	
linear_constraint = LinearConstraint([[1, 1]], [0], [1])
bounds = Bounds([0, 0], [1.0, 1.0])

#Defining matrixes where we store results
Hf=np.empty((101,101))
Hnf=np.empty((101,101))

#Solving for Hf,Hnf
for i,beta in enumerate(np.linspace(0,1,101)):
	for j,ctw in enumerate(np.linspace(0,1,101)):
		try:
			res = minimize(ob_func32, [0.5,0.5], method='trust-constr',
               constraints=[linear_constraint],
               options={'verbose': 1}, bounds=bounds)
			Hf[i,j]=res.x[0]
			Hnf[i,j]=res.x[1]
		except:
			pass
#When solving for beta=0.05, c=0.96 the solver does not provide a result, hence I interpolate with the surrounding values	
Hf[5,86]=0.54	
Hnf[5,86]=0.45863

#### GRAPHS ####
#H
fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), np.add(Hf,Hnf),100)
fig.colorbar(cp) 
ax.set_title('Optimal H')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), np.add(Hf,Hnf),100,colors='black')
ax.clabel(CS, inline=1, fontsize=4)
ax.set_title('Optimal H')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")
np.where(np.add(Hf,Hnf)>=0.99)
#Hf
fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), Hf,10)
fig.colorbar(cp) 
ax.set_title('Optimal Hf')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), Hf,5,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Optimal Hf')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")
#Hnf
fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), Hnf,10)
fig.colorbar(cp) 
ax.set_title('Optimal Hnf')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), Hnf,5,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Optimal Hnf')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")

#Hf/H
fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), np.divide(Hf,np.add(Hf,Hnf)),10)
fig.colorbar(cp) 
ax.set_title('Optimal Hf/H')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), np.divide(Hf,np.add(Hf,Hnf)),10,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Optimal Hf/H')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")

#Y
Y=np.empty((101,101))
for i in range(101):
	for j in range(101):
		Y[i,j]=(mtest.Af*Hf[i,j]**((mtest.rho-1)/mtest.rho)+j/100*mtest.Anf*Hnf[i,j]**((mtest.rho-1)/mtest.rho))**(mtest.rho/(mtest.rho-1))

fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), Y,10)
fig.colorbar(cp) 
ax.set_title('Optimal Output')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), Y,10,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Optimal Output')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")

#Welfare
W=np.empty((101,101))
for i in range(101):
	for j in range(101):
		W[i,j]=Y[i,j]-mtest.kf*Hf[i,j]-mtest.knf*Hnf[i,j]-mtest.omega*(1-mtest.gamma)*i/100*(Hf[i,j]**2)*mtest.io/mtest.N
	
fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), W,10)
fig.colorbar(cp) 
ax.set_title('Welfare')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()
	
fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), W,10,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Welfare')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")		

#Infections
I=np.empty((101,101))
for i in range(101):
	for j in range(101):
		I[i,j]=i/100*(Hf[i,j]**2)*mtest.io/mtest.N

fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), I,10)
fig.colorbar(cp) 
ax.set_title('Infections')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), I,10,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Infections')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")		

#Deaths
D=np.empty((101,101))
for i in range(101):
	for j in range(101):
		D[i,j]=(1-mtest.gamma)*i/100*(Hf[i,j]**2)*mtest.io/mtest.N

fig,ax=plt.subplots()
cp = ax.contourf(np.linspace(0,1,101), np.linspace(0,1,101), D,10)
fig.colorbar(cp) 
ax.set_title('Deaths')
ax.set_xlabel('c(TW)')
ax.set_ylabel('β(HC)')
plt.show()

fig, ax = plt.subplots()
CS = ax.contour(np.linspace(0,1,101), np.linspace(0,1,101), D,10,colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Deaths')
plt.ylabel("β(HC)")
plt.xlabel("c(TW)")		



