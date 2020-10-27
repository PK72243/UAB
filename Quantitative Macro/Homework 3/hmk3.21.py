import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint,Bounds,minimize,fsolve

################# PART WITH CONTIONOUS STATE SPACE K AND CONTIONOUS LABOUR CHOICE
#parameters
kappa=5.24
v=2
alpha=0.679
delta=0.013
beta=0.988
epsilon_V=3	# I set 2 epsilons: ine for thetas for Value function
epsilon_l=0.01	# and one for the labour
#knots parameters
a_k=0.2
b_k=60
m_k=50

#Utility function
def u(c,h,kappa=kappa,v=v):
	return np.log(c)-kappa*((h**(1+1/v))/(1+1/v))
#spline of order one
def spline(j,k_arg,k_vec):
	if (j==0):
		if(k_arg==k_vec[j]):
			res=1
		elif (k_vec[j]<k_arg<=k_vec[j+1]):
			res=(k_vec[j+1]-k_arg)/(k_vec[j+1]-k_vec[j])
		else: 
			res=0
	elif (j==len(k_vec)-1):
		if(k_arg==k_vec[j]):
			res=1
		elif (k_vec[j-1]<=k_arg<=k_vec[j]):
			res=(k_arg-k_vec[j-1])/(k_vec[j]-k_vec[j-1])
		else:
			res=0
	else:
		if (k_vec[j-1]<=k_arg<=k_vec[j]):
			res=(k_arg-k_vec[j-1])/(k_vec[j]-k_vec[j-1])
		elif (k_vec[j]<k_arg<=k_vec[j+1]):
			res=(k_vec[j+1]-k_arg)/(k_vec[j+1]-k_vec[j])
		else:
			res=0
	return res
#spline approximation
def spline_aprox(theta,k_arg,k_vec):
	res=0
	for i in range(len(theta)):
		res_i=theta[i]*spline(i,k_arg,k_vec)
		res=res+res_i
	return res
# two functions we use to find thetas (step 5 from notes; from the LHS we subtract RHS)
def coef_V(thetas):
	res=[]
	for i in range(m_k):
		res=np.append(res,V[i]-spline_aprox(thetas,k[i],k))
	return res
def coef_l(thetas):
	res=[]
	for i in range(m_k):
		res=np.append(res,gl[i]-spline_aprox(thetas,k[i],k))
	return res

start_time=time.time()
k=np.linspace(a_k,b_k,m_k)		#knots for k
theta_v=[np.ones(m_k)] 	#initial guess for thetas for value function
theta_l=[np.ones(m_k)] 	#initial guess for thetas for labour
gk=[np.zeros((m_k,1),dtype='float64')]	#Matrix for storing policy function for capital
gc=[np.zeros((m_k,1),dtype='float64')] #Matrix for storing policy function for consumption
distance_V=np.array([epsilon_V+1],dtype='float64') #initialze a distance for value function with 1+epsilon, so it passes through the first loop condition
distance_l=np.array([epsilon_l+1],dtype='float64') #initialze a distance for thetas with 1+epsilon, so it passes through the first loop condition
V=np.zeros((m_k,1),dtype='float64')	#auxilliary matrix for storing value function
gl=np.zeros((m_k,1),dtype='float64') # auxiliary matrix for storing policy function for labour
counter=0

while((distance_V[counter]>epsilon_V or distance_l[counter]>epsilon_l)):	#loop for iterating
	gk.append(np.zeros((m_k,1),dtype='float64'))		#initialize policy function for capital in the next period
	gc.append(np.zeros((m_k,1),dtype='float64'))	#initialze policy function for consumption in the next period
	theta_v.append(np.zeros((m_k,1),dtype='float64'))	#initialze theta_v for consumption in the next period	
	theta_l.append(np.zeros((m_k,1),dtype='float64'))	#initialze theta_l for consumption in the next period
	for i in range(m_k):
		l=spline_aprox(theta_l[counter],k[i],k)		# l approximation for ki
		x=lambda k_j : -(u(k[i]**(1-alpha)*l**(alpha)+k[i]*(1-delta)-k_j,l)+beta*spline_aprox(theta_v[counter],k_j,k))	#utility function + continuation value
		bounds = Bounds(k[0],min(k[i]**(1-alpha)*l**(alpha)+k[i]*(1-delta),k[-1])) #k' must be within the knots and consumption must be positive
		starting_point=0.5*min(k[i]**(1-alpha)*l**(alpha)+k[i]*(1-delta),k[-1])		#choosing a starting point in the middle ok k'
		res=minimize(x,starting_point,bounds=bounds,method='trust-constr')			# Step 3 - looking for the optimal decision rule for capital
		gk[counter+1][i]=res.x		#storing optimal decision rule
		V[i]=u(k[i]**(1-alpha)*l**(alpha)+k[i]*(1-delta)-res.x,l)+beta*spline_aprox(theta_v[counter],res.x,k)	#Step 4 updating value function
		gc[counter+1][i]=k[i]**(1-alpha)*l**(alpha)+(1-delta)*k[i]-res.x 	#optimal capital
		gl[i]=(gc[counter+1][i]*kappa/alpha*k[i]**(alpha-1))**(1/(alpha-1-1/v))	#optimal l - see equation 13 in the report
	init_guess=np.ones((m_k,1))		#initial guess needed to fsolve
	theta_v[counter+1]=fsolve(coef_V,init_guess)	#solving for thetas v
	theta_l[counter+1]=fsolve(coef_l,init_guess)	#solving for thetas l
	distance_V=np.append(distance_V,[(sum(np.subtract(theta_v[counter],theta_v[counter+1])**(2)))**(0.5)]) #distance between two value functions
	distance_l=np.append(distance_l,[(sum(np.subtract(theta_l[counter],theta_l[counter+1])**(2)))**(0.5)]) #distance between two value functions
	counter+=1
VFI_time=time.time()-start_time	#end of timing

spline_aprox(theta_l[-1],0.2,k)

#Policy functions
fig, axs = plt.subplots(1,2,figsize=(21,8))
axs[0].plot(k,gk[-1])
axs[0].set_title("Policy function for $k_{t+1}$",size=22)
axs[0].set_ylabel("$k_{t+1}$",size=16)
axs[0].set_xlabel("$k_{t}$",size=16)
axs[1].plot(k,gc[-1])
axs[1].set_title("Policy function for ct",size=22)
axs[1].set_ylabel("$c_{t}$",size=16)
axs[1].set_xlabel("$k_{t}$",size=16)
plt.show()

grid_freq=1000		#grid density for a plot
x=np.linspace(a_k,b_k,grid_freq)
y=np.zeros((grid_freq,1))
for i in range(grid_freq):
	y[i]=spline_aprox(theta_v[-1],x[i],k)
fig,ax=plt.subplots()
ax.plot(x,y,color='blue')
ax.set_title("Value function")
ax.set_xlabel("$k_{t}$")
plt.show()
z=np.zeros((grid_freq,1))
for i in range(grid_freq):
	z[i]=spline_aprox(theta_l[-1],x[i],k)
fig,ax=plt.subplots()
ax.plot(x,z,color='blue')
ax.set_title("Labour Decision Rule")
ax.set_xlabel("$k_{t}$")
plt.show()
