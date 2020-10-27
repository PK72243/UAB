import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint,Bounds,minimize,fsolve
import time
### For this exercise I use many materials provided by Albert, but only for the one dimension
a=0.2
b=60
m=60
alpha=0.679
delta=0.013
beta=0.988
d=10
epsilon=0.09

# Construct Chebyshev nodes-----------------------
def cheb_nodes2d(m,a,b):
    # n: grid size
    # a,b: upper and lower bound 1st dimension
    x = []
    z = []
    for j in range(1,m+1):   
        z_k=-np.cos(np.pi*(2*j-1)/(2*m))   ## Get the Chebyshed node in [-1,1]
        x_j=(z_k+1)*((b-a)/2)+a  ## Convert z_point to x dimension
        z.append(z_k)
        x.append(x_j)
    return (np.array(z),np.array(x))
# Create the Chebyshev basis functions -------------------
def T(d,x):
    #d: Degree level
    #x: Chebyshev node
    psi = []
    psi.append(np.ones(len(x)))
    psi.append(x)
    for i in range(1,d):
	    p = 2*x*psi[i]-psi[i-1]
	    psi.append(p)
    key = np.matrix(psi[d]) 
    return key   
# Complete the CHebyshev function value -------------------
def T_v(d,x):
    #d: Degree level
    #x: Chebyshev node
    psi = []
    psi.append(1)
    psi.append(x)
    for i in range(1,d):
	    p = 2*x*psi[i]-psi[i-1]
	    psi.append(p)
    key = psi[d]
    return key   
# Obtain coefficients running OLS -------------------------
def coeff(z,y_nodes,d):
    theta=np.empty((d+1,1),'float64')
    theta.shape = (d+1,1)
    for i in range(d+1):  ##loop over first dimension
	     theta[i] = (np.sum(np.array(y_nodes)*np.array(T(i,z).T)))/np.array((T(i,z)*T(i,z).T))
    return theta

# Obtain approximation function for all nodes---------------------------------
def f_approx_node(x,theta,d):
	f = []
	in1 = (2*(x-a)/(b-a)-1)
	for u in range(d+1):
		f.append(np.array(theta[u])*np.array(T(u,in1).T))
	return sum(f)

# Obtain approximation function for x---------------------------------
def f_approx_x(x,theta,d):
	f = []
	in1 = (2*(x-a)/(b-a)-1)
	for u in range(d+1):
		f.append(theta[u]*T_v(u,in1))
	return sum(f)
#utility function
def u(c):
	return np.log(c)
#-----------------------------------------------------------------
start_time=time.time()
z,x=cheb_nodes2d(m,a,b)       #creating Chebyshev nodes
theta=[np.ones((d+1,1),dtype='float64')]	#initial guess for thetas for value function
gk=[np.zeros((m,1),dtype='float64')]	#Matrix for storing policy function for capital
gc=[np.zeros((m,1),dtype='float64')] #Matrix for storing policy function for consumption
distance=np.array([epsilon+1],dtype='float64') #initialze a distance for value function with 1+epsilon, so it passes through the first loop condition
V=[np.zeros((m,1),dtype='float64')]	#auxilliary matrix for storing value function
counter=0
while(distance[counter]>epsilon):	#loop for iterating
	V.append(np.zeros((m,1),dtype='float64'))	#initialze value function in the next period
	gk.append(np.zeros((m,1),dtype='float64'))	#initialize policy function for capital in the next period
	gc.append(np.zeros((m,1),dtype='float64'))	#initialze policy function for consumption in the next period
	theta.append(np.zeros((m,1),dtype='float64'))	#initialze thetas in the next period
	for i in range(m):
		y=lambda x_j : -(u(x[i]**(1-alpha)+x[i]*(1-delta)-x_j)+beta*f_approx_x(x_j,theta[counter],d))	#utility function + continuation value
		bounds = Bounds(x[0],min(x[i]**(1-alpha)+x[i]*(1-delta),x[-1])) #x' must be within the knots and consumption must be positive
		starting_point=0.5*min(x[i]**(1-alpha)+x[i]*(1-delta),x[-1])		#choosing a starting point in the middle ox x'
		res=minimize(y,starting_point,bounds=bounds,method='trust-constr')			# Step 3 - looking for the optimal decision rule for capital
		gk[counter+1][i]=res.x		#storing optimal decision rule
		V[counter+1][i]=u(x[i]**(1-alpha)+x[i]*(1-delta)-res.x)+beta*f_approx_x(res.x,theta[counter],d)	#Step 4 updating value function
		gc[counter+1][i]=x[i]**(1-alpha)*+(1-delta)*x[i]-res.x 	#optimal capital
		
	theta[counter+1]=coeff(z,V[counter+1],d)	#solving for thetas v
	distance=np.append(distance,[(sum(np.subtract(theta[counter],theta[counter+1])**(2)))**(0.5)]) #distance between two value functions
	counter+=1
VFI_time=time.time()-start_time	#end of timing

#Policy functions plots
fig, axs = plt.subplots(1,2,figsize=(21,8))
axs[0].plot(x,gk[-1])
axs[0].set_title("Policy function for $k_{t+1}$",size=22)
axs[0].set_ylabel("$k_{t+1}$",size=16)
axs[0].set_xlabel("$k_{t}$",size=16)
axs[1].plot(x,gc[-1])
axs[1].set_title("Policy function for ct",size=22)
axs[1].set_ylabel("$c_{t}$",size=16)
axs[1].set_xlabel("$k_{t}$",size=16)
plt.show()

grid_freq=1000		#grid density for a plot
g=np.linspace(a,b,grid_freq)
y=np.zeros((grid_freq,1))
for i in range(grid_freq):
	y[i]=f_approx_x(g[i],theta[-2],d)
fig,ax=plt.subplots()
ax.plot(g,y,color='blue')
ax.set_title("Value function")
ax.set_xlabel("$k_{t}$")
plt.show()
