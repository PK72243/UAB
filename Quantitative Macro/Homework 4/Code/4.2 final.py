from scipy.optimize import Bounds,minimize,minimize_scalar
from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import bernoulli,norm
from numpy.random import multinomial
import matplotlib.pyplot as plt
import quantecon as qe
import pandas as pd
import seaborn as sns
# %%#parameters
rho=0.06
beta=1/(1+rho)
sigma=2
sigma_y=0.3  
gamma=0.5     
c_bar=100
alpha=0.33
delta=0.025
utility_quadratic=False
a_bar_zero=True
# %%functions
def quad_u(c):
	if c<0:
		u=-999999999
	else:
		u=-0.5*(c-c_bar)**2
	return u
def crra_u(c):
	if c<=0:
		u=-99999999
	else:
		if sigma==1:
			u=np.log(c)
		else:
			u=(c**(1-sigma)-1)/(1-sigma)
	return u

def u(c,preferences_quadratic=utility_quadratic):     #function for preferences
	if preferences_quadratic==True:
		u=quad_u(c)
	else:
		u=crra_u(c)
	return u
def uc(c,preferences_quadratic=utility_quadratic):     #function for marginal utility
	if preferences_quadratic==True:
		uc=c_bar-c
	else:
		if c<=0:
			u=99999999999999999999999
		else:
			u=c**(-sigma)
	return u

def a_bar(r,y_min,zero=a_bar_zero):      #function for borrowing constraint 
	if zero==True:
		a=0
	else:
		a=1/(2+r)*y_min
	return a
#To create non-even spaced grids
def my_lin(lb, ub, steps, spacing=2):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)])

# delivers index of the element in array closest to value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def random_coin(p):    # Drawing a coin, useful for Markov processes
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
#Auxiliary function for drawing a future T states
def future_state(y,pi,T):
	# for an index of y and transition matrix it draws an index of the future state
	X = np.empty(T, dtype=int)
	n=len(pi)
	P_dist = [np.cumsum(pi[i, :]) for i in range(n)]
	X[0] = y
	for t in range(T-1):
		       X[t+1] = qe.random.draw(P_dist[X[t]])
	return X
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    # Found on the web https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def plot_distribution(state, state_name, save=False,bins=20):# Modified Albert's function
    fig, ax = plt.subplots()
    sns.distplot(state, label=state_name,bins=bins)
    plt.title('Distribution of '+state_name)
    plt.xlabel(state_name)
    plt.ylabel("Density")
    plt.legend()
    if save==True:
        fig.savefig('distribution'+state_name+'.png')
    return plt.show()
def tauchen(rho, sigma_u, m=3, n=7): # it is a function feom quantecon module, but I have some problems running it, so I define it here
    """
    Computes the Markov matrix associated with a discretized version of
    the linear Gaussian AR(1) process

        y_{t+1} = rho * y_t + u_{t+1}

    according to Tauchen's method.  Here {u_t} is an iid Gaussian
    process with zero mean.

    Parameters
    ----------
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    x : array_like(float, ndim=1)
        The state space of the discretized process
    P : array_like(float, ndim=2)
        The Markov transition matrix where P[i, j] is the probability
        of transitioning from x[i] to x[j]

    """
    F = norm(loc=0, scale=sigma_u).cdf

    # standard deviation of y_t
    std_y = np.sqrt(sigma_u**2 / (1-rho**2))

    # top of discrete state space
    x_max = m * std_y

    # bottom of discrete state space
    x_min = - x_max

    # discretized state space
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    for i in range(n):
        P[i, 0] = F(x[0]-rho * x[i] + half_step)
        P[i, n-1] = 1 - F(x[n-1] - rho * x[i] - half_step)
        for j in range(1, n-1):
            z = x[j] - rho * x[i]
            P[i, j] = F(z + half_step) - F(z - half_step)

    return x, P
			
def table_kmr(data):#this function creates a table similar to the KMR one
	index=["Mean","Q1","Q2","Q3","Q4","Q5","90-95","95-99","Top 1%","Gini"]
	mean=np.mean(data)
	q1=np.quantile(data,0.2)
	q2=np.quantile(data,0.4)
	q3=np.quantile(data,0.6)
	q4=np.quantile(data,0.8)
	q90=np.quantile(data,0.90)
	q95=np.quantile(data,0.95)
	q99=np.quantile(data,0.99)
	sum_data=np.sum(data)
	
	d1=np.sum(data[np.where(data<q1)])/sum_data
	d2=np.sum(data[np.where(data<q2)])/sum_data-d1
	d3=np.sum(data[np.where(data<q3)])/sum_data-d2-d1
	d4=np.sum(data[np.where(data<q4)])/sum_data-d3-d2-d1
	d5=1-d1-d2-d3-d4
	d9095=np.sum(data[np.where(data<q95)])/sum_data-np.sum(data[np.where(data<q90)])/sum_data
	d9599=np.sum(data[np.where(data<q99)])/sum_data-np.sum(data[np.where(data<q95)])/sum_data
	d99=1-np.sum(data[np.where(data<q99)])/sum_data
	gini=gini_coefficient(data)
	return pd.DataFrame([mean,100*d1,100*d2,100*d3,100*d4,100*d5,100*d9095,100*d9599,100*d99,gini],index=index)


#%% Function for solving infinite HH's problem
    
def hh_cont(r,N_y=2,rho_y=0.9,epsilon=0.2,N_a=30,a_max=60,method='slinear'):
	"""
	

	Parameters
	----------
	r : interest rate
	epsilon : tolerance. The default is 2.
	rho_y: the autocorrelation coefficient of the AR1 process
	N_a : number of gridpoints for a. The default is 60.
	N_y : number of gridpoints for y. The default is 2.
	a_max : Maximum asset level. The default is 60.
	method : method for interpolation. The default is 'slinear'.


	"""
	a_grid=my_lin(-a_bar(r,1-sigma_y),a_max,N_a)
	
	#creating grid for y and transition matrix
	if N_y==2:
		y_grid=[1-sigma_y,1+sigma_y]
		pi=np.empty((N_y,N_y),dtype='float64')
		pi[0,0],pi[1,1]=(1+gamma)/2,(1+gamma)/2
		pi[0,1],pi[1,0]=(1-gamma)/2,(1-gamma)/2
	else:
		y_grid,pi=tauchen(rho_y,sigma_y,m=3,n=N_y)
		y_grid=y_grid+np.abs(y_grid[0])+0.1
	
	# Set empty Value and Policy functions: 
	g_a = [np.empty((N_a, N_y),dtype='float64')]
	g_c = [np.empty((N_a, N_y),dtype='float64')]
		## Initial Guess
	V = [np.zeros((N_a, N_y),dtype='float64')]
	distance=np.array([epsilon+1],dtype='float64') #initialze a distance with 1+epsilon, so it passes through the first loop condition
	
	
	
	w=(1-alpha)*((delta+r)/alpha)**(alpha/(alpha-1))
	counter=0

	while(distance[counter]>epsilon):
		V.append(np.zeros((N_a,N_y),dtype='float64'))	#initialze value function in the next period
		g_a.append(np.zeros((N_a,N_y),dtype='float64'))		#initialize policy function for capital in the next period
		g_c.append(np.zeros((N_a,N_y),dtype='float64'))	#initialze policy function for consumption in the next period
		
		def cont_val(a_prime,l):# expected value of the continuation value for value function V for an a' an current state l
			v=0
			for l_prime in range(N_y):
				v_func = interp1d(a_grid, V[counter][:,l_prime], fill_value='extrapolate', kind=method)
				v=v+v_func(a_prime)*pi[l,l_prime]
			return v
		
		for i in range(N_a):
			for l in range(N_y):
		
				def criterion_func(a_prime):  ## on a_prime 
					c=a_grid[i]*(1+r)+w*y_grid[l]-a_prime
					return -(u(c) + beta*cont_val(a_prime,l))
			
				a_max=min(a_grid[i]*(1+r)+w*y_grid[l],a_grid[-1])
				a_min=-a_bar(r, y_grid[0])
				res=minimize_scalar(criterion_func, bounds=(a_min, a_max), method='bounded')
				g_a[counter+1][i,l]=res.x
				g_c[counter+1][i,l]=w*y_grid[l]+(1+r)*a_grid[i]-res.x
				V[counter+1][i,l]=-res.fun  
		distance=np.append(distance,[sum((sum(np.subtract(V[counter],V[counter+1])**(2))))**(0.5)]) #distance between two value functions
		counter+=1
	return V[-1],g_a[-1],g_c[-1]
	 
#%%		Stationary invariant distribution                                                                              
def phi_dist(V_func,a_policy,c_policy,N_y=2,rho_y=0.9,a_1=0,N=8000,T=250,N_a=30,a_max=60,method='slinear'):
	"""
	

	Parameters
	----------
	V_func : Value function.
	a_policy : POlicy function for assets.
	c_policy : POlicy function for consumption.
	
	rho_y: the autocorrelation coefficient of the AR1 process
	a_1 : initial asset level. The default is 10.
	N : Number of simulations. The default is 10000.
	T : Number of time periods. The default is 200.
	N_a : number of gridpoints for a. The default is 60.
	N_y : number of gridpoints for y. The default is 2.
	a_max : Maximum asset level. The default is 60.
	method : method for interpolation. The default is 'slinear'.


	"""
	a_grid=my_lin(0,a_max,N_a)
	#creating grid for y and transition matrix
	if N_y==2:
		y_grid=[1-sigma_y,1+sigma_y]
		pi=np.empty((N_y,N_y),dtype='float64')
		pi[0,0],pi[1,1]=(1+gamma)/2,(1+gamma)/2
		pi[0,1],pi[1,0]=(1-gamma)/2,(1-gamma)/2
		pini=[0.5,0.5]
	else:
		y_grid,pi=tauchen(rho_y,sigma_y,m=3,n=N_y)
		y_grid=y_grid+np.abs(y_grid[0])+0.1
		pini=np.ones((N_y,1))/N_y
		
	a = np.empty((T,N))
	y_state = np.empty((T,N))
	c = np.empty((T,N))
	V = np.empty((T,N))
	mean_a = np.empty(T)
	var_a = np.empty(T)
	
	g_a_functions={}
	g_c_functions={}
	V_functions={}
	for i_l in range(N_y):
			g_a_functions[i_l] = interp1d(a_grid, a_policy[:,i_l], fill_value="extrapolate",kind=method)
			V_functions[i_l] = interp1d(a_grid, V_func[:,i_l], fill_value="extrapolate",kind=method)
			g_c_functions[i_l] = interp1d(a_grid, c_policy[:,i_l], fill_value="extrapolate",kind=method)
		
	
	#Drawing income states with initial distribution pini and initial assets
	for i in range(N):
		y_state[0,i]=np.array(np.where(np.random.multinomial(1,pini,1)[0]==1)).flatten()
		y_state[1:,i]=future_state(y_state[0,i],pi,T-1)
	#staeting with 10 assets
	a[0,:]=a_1
	
	for t in range(1,T):
		for n in range(0,N):
			a[t,n]=g_a_functions[y_state[t,n]](a[t-1,n])
			c[t,n]=g_c_functions[y_state[t,n]](a[t,n])
			V[t,n]=V_functions[y_state[t,n]](a[t,n])
		mean_a[t] = np.mean(a[t,:])
		var_a[t]  = np.var(a[t,:])
		#if t%10 == 0:
			#print(t)
	return a[-1,:],c[-1,:],V[-1,:],y_state[-1,:],mean_a,var_a
#%%Solving for GE
r_init=0.04 #initial guess
omega=0.0004 #step parameter for updating guess
epsilon=0.015 #tolerance for distance function
distance=100 #initial distance such it passes the first while loop
r=r_init
while np.abs(distance)>epsilon:
	V,g_a,g_c=hh_cont(r)	
	print("HH's solved")
	a,c,V,y,mean,var=phi_dist(V,g_a,g_c ) 
	print("distribution computed")
	ea=np.mean(a)
	k=((delta+r)/alpha)**(1/(alpha-1))
	distance=k-ea
	new_r=r+omega*distance
	r=new_r
	print('New interest rate:',r,"Distance",distance)
#%%saving
np.save('r',r)
np.save('a',a)
np.save('c',c)
np.save('V',V)
np.save('y',y)	
#%% Aiyagari graph
grid1=np.array([-0.02,0,0.02,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.08,0.09])
ear=np.zeros((len(grid1)))
for i in range(len(grid1)):
	a,b,c=hh_cont(grid1[i])
	d,e,f,g,h,j=phi_dist(a,b,c)
	ear[i]=np.mean(d)
	print(i)
	
#np.save('ear',ear)
#ear=np.load('ear.npy)
grid2=my_lin(-delta+0.005,0.09,100,spacing=1)
kr=np.zeros((len(grid2)))
for i in range(len(grid2)):
	kr[i]=k=((delta+grid2[i])/alpha)**(1/(alpha-1))
	

ear_func=interp1d(grid1,ear,kind='slinear')

fig,ax=plt.subplots()
ax.plot(ear_func(grid2),grid2,label="Ea(r)")
ax.plot(kr,grid2,label="K(r)")
plt.xlabel("Assets, Capital")
plt.ylabel("interest rate")
plt.title("Steady-state determination")
ax.set_ylim([-0.03,0.1])
plt.legend()
plt.show()
#%% Loading of results
a=np.load('a.npy')
r=np.load('r.npy')
c=np.load('c.npy')
V=np.load('V.npy')
y=np.load('y.npy')
#%%Comparison of distributions

tab=pd.merge(table_kmr(a),table_kmr(c),left_index=True,right_index=True,how="outer")
tab=pd.merge(tab,table_kmr(V),left_index=True,right_index=True,how="outer")
tab.columns=["Assets","Consumption","Value function"]
print(tab.to_latex())

plot_distribution(V, 'Value function', save=True)  
plot_distribution(a, 'Assets', save=True)  
plot_distribution(c, 'Consumption', save=True)  

#%% AIYAGARI
beta=0.96
alpha=0.36
delta=0.1
sigma=3
sigma_y=0.4
rho_y=0.9

r_init=0.04 #initial guess
omega=0.0004 #step parameter for updating guess
epsilon=0.22 #tolerance for distance function
distance=100 #initial distance such it passes the first while loop
r=r_init
#%%
while np.abs(distance)>epsilon:
	V,g_a,g_c=hh_cont(r,N_y=7,rho_y=rho_y)	
	print("HH's solved")
	a,c,V,y,mean,var=phi_dist(V,g_a,g_c,N_y=7,rho_y=rho_y) 
	print("distribution computed")
	ea=np.mean(a)
	k=((delta+r)/alpha)**(1/(alpha-1))
	distance=k-ea
	new_r=r+omega*distance
	r=new_r
	print('New interest rate:',r,"Distance",distance)
#%%saving
np.save('rAiy',r)
np.save('aAiy',a)
np.save('cAiy',c)
np.save('VAiy',V)
np.save('yAiy',y)
#%% Aiyagari graph
grid1=np.array([-0.02,0,0.02,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.08,0.09])
ear=np.zeros((len(grid1)))
for i in range(len(grid1)):
	a,b,c=hh_cont(grid1[i],N_y=7,rho_y=rho_y)
	d,e,f,g,h,j=phi_dist(a,b,c,N_y=7,rho_y=rho_y)
	ear[i]=np.mean(d)
	print(i)
	
#np.save('ear',ear)
#ear=np.load('ear.npy)

kr=np.zeros((len(grid1)))
for i in range(len(grid1)):
	kr[i]=k=((delta+grid1[i])/alpha)**(1/(alpha-1))
	
grid2=my_lin(-delta+0.01,0.1,100,spacing=1)
ear_func=interp1d(grid1,ear,kind='quadratic')
kr_func=interp1d(grid1,kr,kind='quadratic')

fig,ax=plt.subplots()
ax.plot(ear_func(grid2),grid2,label="Ea(r)")
ax.plot(kr_func(grid2),grid2,label="K(r)")
plt.xlabel("Assets, Capital")
plt.ylabel("interest rate")
plt.title("Steady-state determination")
plt.legend()
plt.show()
#%% Loading of results
a=np.load('aAiy.npy')
r=np.load('rAiy.npy')
c=np.load('cAiy.npy')
V=np.load('VAiy.npy')
y=np.load('yAiy.npy')
#%%Distribution comparison
tab=pd.merge(table_kmr(a),table_kmr(c),left_index=True,right_index=True,how="outer")
tab=pd.merge(tab,table_kmr(V),left_index=True,right_index=True,how="outer")
tab.columns=["Assets","Consumption","Value function"]
print(tab.to_latex())

plot_distribution(V, 'Value function Aiyagari', save=True)  
plot_distribution(a, 'Assets Aiyagari', save=True)  
plot_distribution(c, 'Consumption Aiyagari', save=True)  
	