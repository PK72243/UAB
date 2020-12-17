import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import Bounds,minimize,minimize_scalar
from scipy.interpolate import interp1d
import quantecon as qe
from scipy.stats import bernoulli
from numpy.random import multinomial
# %%#parameters
rho=0.06
beta=1/(1+rho)
r=0.04
w=1
sigma=2
sigma_y=0.1
gamma=0  
c_bar=100
epsilon=0.3
utility_quadratic=True
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
def plot_policy_2d(grid, policy, policy_name,line_45=False):
	fig,ax = plt.subplots()
	if policy_name=="assets":
		for i in range(N_y):
			ax.plot(grid, a_grid[policy[-1][:,i]], label = 'Income '+str(y_grid[i]))
	else:
		for i in range(N_y):
			ax.plot(grid, policy[-1][:,i], label = 'Income '+str(y_grid[i]))
	if line_45 == True:
		ax.plot(grid,grid, linestyle='dashed', label='45 line')
	ax.set_xlabel('Cash on hand')
	ax.legend()
	ax.set_title("Policy function for "+policy_name)
	plt.show() 
def plot_policy_2d2(grid, policy, policy_name, line_45=False):
        
    fig,ax = plt.subplots()
    for i_l in range(N_y):
        ax.plot(grid, policy[:,i_l], label = 'Income  '+str(y_grid[i_l]))
    if line_45 == True:
        ax.plot(grid,grid, linestyle='dashed', label='45 line')
    ax.set_xlabel('Assets today')
    ax.legend()
    ax.set_title(policy_name)
    plt.show() 

# %% VFI discrete
#grids
#grid parameters
N_a, N_y = 500, 2
a_max=60
a_grid=my_lin(-a_bar(r,1-sigma_y),a_max,N_a)
#a_grid=np.linspace(-a_bar(r,1-sigma_y),a_max,N_a)
y_grid=[1-sigma_y,1+sigma_y]
#transition matrix
pi=np.empty((N_y,N_y),dtype='float64')
pi[0,0],pi[1,1]=(1+gamma)/2,(1+gamma)/2
pi[0,1],pi[1,0]=(1-gamma)/2,(1-gamma)/2

## Initial Guess
start_time=time.time() #time stopper
W=np.zeros((N_y,N_a),dtype='float64')	 #initialize a matrix of W with zeros everywhere, slide 33
X=np.zeros((N_y*N_a,N_a),dtype='float64')	 #initialize a matrix of X with zeros everywhere, slide 34
V=[np.zeros((N_a,N_y),dtype='float64')]	#initial guess for the value function, as you can notice I decided to store value of each iteration for deeeper
# analysis, that is the reason why I create here and below a list of lists
ga=[np.zeros((N_a,N_y),dtype='int')]	#initialze policy function for k, here I store indexes of k not values, because than the algorithm in my case is faster
gc=[np.zeros((N_a,N_y),dtype='float64')] #initialze a policy function for c
distance=np.array([epsilon+1],dtype='float64') #initialze a distance with 1+epsilon, so it passes through the first loop condition

return_m = np.empty((N_a*N_y,N_a))
for i in range(N_a):
	for j in range(N_a):
		for l in range(N_y):
			return_m[N_a*(l)+i,j]=u(w*y_grid[l]+(1+r)*a_grid[i]-a_grid[j])



counter=0	#counter for iteration

while(distance[counter]>epsilon):	#loop for iterating
	V.append(np.zeros((N_a,N_y),dtype='float64'))	#initialze value function in the next period
	ga.append(np.zeros((N_a,N_y),dtype='int'))		#initialize policy function for capital in the next period
	gc.append(np.zeros((N_a,N_y),dtype='float64'))	#initialze policy function for consumption in the next period
	for q in range(N_y):
		for p in range(N_a):
			W[q,p]=np.sum(pi[q,:]*V[counter][p,:])
	for i in range(N_a):
		for j in range(N_a):
			for l in range(N_y):
				X[N_a*(l)+i,j]=return_m[N_a*(l)+i,j]+beta*W[l,j]	#X matrix is calculated
	for i in range(N_a):
		for l in range(N_y):
			V[counter+1][i,l]=max(X[N_a*(l)+i,:])
			ga[counter+1][i,l]=np.argmax(X[N_a*(l)+i,:])
			gc[counter+1][i,l]=w*y_grid[l]+(1+r)*a_grid[i]-a_grid[ga[counter+1][i,l]]
	distance=np.append(distance,[sum((sum(np.subtract(V[counter],V[counter+1])**(2))))**(0.5)]) #distance between two value functions
	counter+=1
VFI_time=time.time()-start_time	#end of timing
vfunc_dis=V[-1]
afunc_dis=a_grid[ga[-1]]
cfunc_dis=gc[-1]

plot_policy_2d(a_grid, ga, "assets",line_45=True)		
plot_policy_2d(a_grid, gc, "consumption")	

fig,ax=plt.subplots()
ax.plot(a_grid,V[-1][:,0],label="income {}".format(y_grid[0]))
ax.plot(a_grid,V[-1][:,1],label="income {}".format(y_grid[0]))
ax.set_title("Value function with monotonicity")
ax.set_xlabel("$a_{t}$")
plt.legend()
plt.show()		

# %% VFI continuos
N_a, N_y = 60, 2
a_max=60
a_grid=my_lin(-a_bar(r,1-sigma_y),a_max,N_a)
#a_grid=np.linspace(-a_bar(r,1-sigma_y),a_max,N_a)
y_grid=[1-sigma_y,1+sigma_y]
#transition matrix
pi=np.empty((N_y,N_y),dtype='float64')
pi[0,0],pi[1,1]=(1+gamma)/2,(1+gamma)/2
pi[0,1],pi[1,0]=(1-gamma)/2,(1-gamma)/2
epsilon=2


# Set empty Value and Policy functions: ===
g_a = [np.empty((N_a, N_y),dtype='float64')]
g_c = [np.empty((N_a, N_y),dtype='float64')]
## Initial Guess
V = [np.zeros((N_a, N_y),dtype='float64')]
distance=np.array([epsilon+1],dtype='float64') #initialze a distance with 1+epsilon, so it passes through the first loop condition

method='slinear'
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
			                                                                                   
  
vfunc_con=V[-1]
afunc_con=g_a[-1]
cfunc_con=g_c[-1]
if utility_quadratic==True:# it is used only to prepare some graphs
	vfunc_con1=V[-1]
	afunc_con1=g_a[-1]
	cfunc_con1=g_c[-1]
else:
	vfunc_con2=V[-1]
	afunc_con2=g_a[-1]
	cfunc_con2=g_c[-1]
#%%VFI continous plots

N_A_large = 200
A_GRID =  np.linspace(-a_bar(r,1-sigma_y),a_max, N_A_large)
V_NEW = np.empty((N_A_large, N_y))
G_A = np.empty((N_A_large, N_y))
G_C = np.empty((N_A_large, N_y))
               
for i_l in range(0,N_y):
    v_func = interp1d(a_grid, vfunc_con[:,i_l], fill_value="extrapolate",kind=method)
    g_a_func  = interp1d(a_grid, afunc_con[:,i_l], fill_value="extrapolate",kind=method)
    g_c_func = interp1d(a_grid, cfunc_con[:,i_l], fill_value="extrapolate",kind=method)
    
    V_NEW[:,i_l] = v_func(A_GRID)
    G_A[:,i_l]  = g_a_func(A_GRID)
    G_C[:,i_l]  = g_c_func(A_GRID)


plot_policy_2d2(grid=A_GRID, policy=V_NEW, policy_name='Value_function')
plot_policy_2d2(grid=A_GRID, policy=G_A, line_45=True, policy_name='Assets_Policy') 
plot_policy_2d2(grid=A_GRID, policy=G_C, policy_name='Consumption Policy')

#%% Path simulator
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


def path_sim(a_0,y_0,g_a,V,pi=pi,a_grid=a_grid,y_grid=y_grid,T=45):
	# Function for simulating paths for a single agent
	#a_o, y_0 values not indexes
	if y_0 in y_grid:
		g_a_functions={}
		#g_c_functions={}
		V_functions={}
		for i_l in range(N_y):
			g_a_functions[i_l] = interp1d(a_grid, g_a[:,i_l], fill_value="extrapolate",kind='quadratic')
			V_functions[i_l] = interp1d(a_grid, V[:,i_l], fill_value="extrapolate",kind='quadratic')
			#g_c_functions[i_l] = interp1d(a_grid, g_c[:,i_l], fill_value="extrapolate",kind='quadratic')
		y_path=future_state(find_nearest(y_grid,y_0),pi,T+1)
		a_path,c_path,v_path=np.empty((T+1)), np.empty((T+1)), np.empty((T+1))
		c_path[0],v_path[0],a_path[0] = 0,0,a_0
		for i in range(0,T):
			a_path[i+1]=g_a_functions[y_path[i]](a_path[i])
			c_path[i]=a_path[i]*(1+r)+w*y_grid[y_path[i]]-a_path[i+1]
			v_path[i]=V_functions[y_path[i]](a_path[i])
		c_path[T]=a_path[T]*(1+r)+w*y_grid[y_path[T]]-g_a_functions[y_path[i]](a_path[T])
		v_path[T]=V_functions[y_path[T]](a_path[T])
		return y_path, a_path, c_path, v_path
	else:
		raise Warning("Wrong y_0")
	
#y_path, a_path, c_path, v_path = path_sim(0,0.7,afunc_con,vfunc_con)	

def path_sim2(a_0,y_0,g_a,V,N,pi=pi,a_grid=a_grid,y_grid=y_grid,T=45):
	y_mean_path, a_mean_path, c_mean_path, v_mean_path = np.empty((T+1,1)), np.empty((T+1,1)), np.empty((T+1,1)), np.empty((T+1,1))
	y_path, a_path, c_path, v_path =np.empty((T+1,N)), np.empty((T+1,N)), np.empty((T+1,N)), np.empty((T+1,N))
	for n in range(N):
		y_path[:,n], a_path[:,n], c_path[:,n], v_path[:,n] = path_sim(a_0,y_0,g_a,V,pi=pi,a_grid=a_grid,y_grid=y_grid,T=45)
	y_mean_path=np.mean(y_path,axis=1)
	a_mean_path=np.mean(a_path,axis=1)
	c_mean_path=np.mean(c_path,axis=1)
	v_mean_path=np.mean(v_path,axis=1)
	return y_mean_path, a_mean_path,c_mean_path,v_mean_path
		
#y_mean_path, a_mean_path, c_mean_path, v_mean_path = path_sim2(0,0.7,afunc_con,vfunc_con,N=10000)	
	
#%%%%
#Finite horizon
N_a, N_y, T = 60, 2, 45
a_max=60
a_grid=my_lin(-a_bar(r,1-sigma_y),a_max,N_a)
#a_grid=np.linspace(-a_bar(r,1-sigma_y),a_max,N_a)
y_grid=[1-sigma_y,1+sigma_y]
#transition matrix
pi=np.empty((N_y,N_y),dtype='float64')
pi[0,0],pi[1,1]=(1+gamma)/2,(1+gamma)/2
pi[0,1],pi[1,0]=(1-gamma)/2,(1-gamma)/2

t_grid=np.arange(T+1)
t=0
gc_fin, ga_fin, V_fin, vc_fin = np.zeros((N_a,N_y,T+1)), np.zeros((N_a,N_y,T+1)), np.zeros((N_a,N_y,T+1)), np.zeros((N_a,N_y,T+1))


for t in reversed(range(T+1)):
	if t==T:
		for x in range(N_a):
			for l in range(N_y):
				gc_fin[x,l,t]=a_grid[x]*(1+r)+w*y_grid[l]
				V_fin[x,l,t]=u(gc_fin[x,l,t])
				#Vc_fin[x,l,t]=uc(gc_fin[x,l,t])
				
		ga_fin[:,:,t]=0
	elif t<T:
		 for x in range(N_a):
			 for l in range(N_y):
				 def cont_val2(a_prime,l):# expected value of the continuation value for value function V for an a' an current state l
					 v=0
					 for l_prime in range(N_y):
						 v_func = interp1d(a_grid, V_fin[:,l_prime,t+1], fill_value='extrapolate', kind='quadratic')
						 v=v+v_func(a_prime)*pi[l,l_prime]*beta
					 return v
				 def criterion_func(a_prime):  ## on a_prime 
					 c=a_grid[x]*(1+r)+w*y_grid[l]-a_prime
					 return -(u(c) + beta*cont_val2(a_prime,l))
				 
				 a_max=min(a_grid[x]*(1+r)+w*y_grid[l],a_grid[-1])
				 a_min=-a_bar(r, y_grid[0])
				 res=minimize_scalar(criterion_func, bounds=(a_min, a_max), method='bounded')
				 ga_fin[x,l,t]=res.x
				 gc_fin[x,l,t]=w*y_grid[l]+(1+r)*a_grid[x]-res.x
				 V_fin[x,l,t]=-res.fun
				 #Vc_fin[x,l,t]=uc(w*y_grid[l]+(1+r)*a_grid[x]-res.x)
		 print(t)
t=5
vfunc_fin=V_fin[:,:,t]
afunc_fin=ga_fin[:,:,t]
cfunc_fin=gc_fin[:,:,t]

if utility_quadratic==True:# it is used only to prepare some graphs
	vfunc_finite1=V_fin
	afunc_finite1=ga_fin
	cfunc_finite1=gc_fin
else:
	vfunc_finite2=V_fin
	afunc_finite2=ga_fin
	cfunc_finite2=gc_fin

#%%# plot for PE task:

N_A_large = 200
A_GRID =  np.linspace(-a_bar(r,1-sigma_y),a_max, N_A_large)
G_C1 = np.empty((N_A_large, N_y))
G_C2 = np.empty((N_A_large, N_y))
    
               

cfunc_fin1=gc_fin[:,:,5]
cfunc_fin2=gc_fin[:,:,40]		
for i_l in range(0,N_y):
    g_c_func1 = interp1d(a_grid, cfunc_fin1[:,i_l], fill_value="extrapolate",kind=method)
    g_c_func2 = interp1d(a_grid, cfunc_fin2[:,i_l], fill_value="extrapolate",kind=method)
    G_C1[:,i_l]  = g_c_func1(A_GRID)
    G_C2[:,i_l]  = g_c_func2(A_GRID)
    
fig,ax = plt.subplots()
ax.plot(A_GRID, G_C1[:,0], label = 'consumption policy for period 5 and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(A_GRID, G_C2[:,0], label = 'consumption policy for period 40 and income '+ str(y_grid[0]))
#ax.plot(A_GRID, G_C1[:,1], label = 'consumption policy for period 5 and income '+ str(y_grid[1]),linestyle='dashed')
#ax.plot(A_GRID, G_C2[:,1], label = 'consumption policy for period 45 and income '+ str(y_grid[1]))
ax.set_xlabel('time')
ax.legend()
ax.set_title('Consumption Policy')
plt.show()    
 
#%% finite paths				 	 
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


def path_sim_fin(a_0,y_0,g_a,V,pi=pi,a_grid=a_grid,y_grid=y_grid,T=T):
	# Function for simulating paths for a single agent
	#a_o, y_0 values not indexes
	if y_0 in y_grid:
		g_a_functions={}
		#g_c_functions={}
		V_functions={}
		for i_l in range(N_y):
			for t in range(T+1):
				g_a_functions[i_l,t] = interp1d(a_grid, g_a[:,i_l,t], fill_value="extrapolate",kind='quadratic')
				V_functions[i_l,t] = interp1d(a_grid, V[:,i_l,t], fill_value="extrapolate",kind='quadratic')
				#g_c_functions[i_l,t] = interp1d(a_grid, g_c[:,i_l,t], fill_value="extrapolate",kind='quadratic')
		
		y_path=future_state(find_nearest(y_grid,y_0),pi,T+1)
		a_path,c_path,v_path=np.empty((T+1)), np.empty((T+1)), np.empty((T+1))
		c_path[0],v_path[0],a_path[0] = 0,0,a_0
		for i in range(0,T):
			a_path[i+1]=g_a_functions[y_path[i],i](a_path[i])
			c_path[i]=a_path[i]*(1+r)+w*y_grid[y_path[i]]-a_path[i+1]
			v_path[i]=V_functions[y_path[i],i](a_path[i])
		c_path[T]=a_path[T]*(1+r)+w*y_grid[y_path[T]]
		v_path[T]=V_functions[y_path[T],T](a_path[T])
		return y_path, a_path, c_path, v_path
	else:
		raise Warning("Wrong y_0")
	
#y_path, a_path, c_path, v_path = path_sim_fin(0,0.7,ga_fin,V_fin)	

def path_sim_fin2(a_0,y_0,g_a,V,N,pi=pi,a_grid=a_grid,y_grid=y_grid,T=T):
	y_mean_path, a_mean_path, c_mean_path, v_mean_path = np.empty((T+1,1)), np.empty((T+1,1)), np.empty((T+1,1)), np.empty((T+1,1))
	y_path, a_path, c_path, v_path =np.empty((T+1,N)), np.empty((T+1,N)), np.empty((T+1,N)), np.empty((T+1,N))
	for n in range(N):
		y_path[:,n], a_path[:,n], c_path[:,n], v_path[:,n] = path_sim_fin(a_0,y_0,g_a,V,pi=pi,a_grid=a_grid,y_grid=y_grid,T=45)
	y_mean_path=np.mean(y_path,axis=1)
	a_mean_path=np.mean(a_path,axis=1)
	c_mean_path=np.mean(c_path,axis=1)
	v_mean_path=np.mean(v_path,axis=1)
	return y_mean_path, a_mean_path,c_mean_path,v_mean_path
		
#y_mean_path, a_mean_path, c_mean_path, v_mean_path = path_sim_fin2(0,0.7,ga_fin,V_fin,N=2000)	
	
#%% PE Extra plots II 4.1
y_1_in, a_1_in, c_1_in, v_1_in = path_sim2(0,1,afunc_con,vfunc_con,N=10000)	
y_2_in, a_2_in, c_2_in, v_2_in = path_sim2(2,1,afunc_con,vfunc_con,N=10000)	
y_3_in, a_3_in, c_3_in, v_3_in = path_sim2(6,1,afunc_con,vfunc_con,N=10000)	
y_4_in, a_4_in, c_4_in, v_4_in = path_sim2(15,1,afunc_con,vfunc_con,N=10000)	


y_1_fin, a_1_fin, c_1_fin, v_1_fin = path_sim_fin2(0,1,ga_fin,V_fin,N=10000)	
y_2_fin, a_2_fin, c_2_fin, v_2_fin = path_sim_fin2(2,1,ga_fin,V_fin,N=10000)	
y_3_fin, a_3_fin, c_3_fin, v_3_fin = path_sim_fin2(6,1,ga_fin,V_fin,N=10000)	
y_4_fin, a_4_fin, c_4_fin, v_4_fin = path_sim_fin2(16,1,ga_fin,V_fin,N=10000)	

fig,ax = plt.subplots()
ax.plot(np.arange(T+1), c_1_in, label = 'assets=0 finite horizon ',linestyle='dashed')
ax.plot(np.arange(T+1), c_2_in, label = 'assets=2 finite horizon ',linestyle='dashed')
ax.plot(np.arange(T+1), c_3_in, label = 'assets=6 finite horizon ',linestyle='dashed')
ax.plot(np.arange(T+1), c_4_in, label = 'assets=15 finite horizon ',linestyle='dashed')
ax.plot(np.arange(T+1), c_1_fin, label = 'assets=0 infinite horizon ')
ax.plot(np.arange(T+1), c_2_fin, label = 'assets=2 infinite horizon ')
ax.plot(np.arange(T+1), c_3_fin, label = 'assets=6 infinite horizon ')
ax.plot(np.arange(T+1), c_4_fin, label = 'assets=15 infinite horizon ')
ax.set_xlabel('time')
ax.legend()
ax.set_title('Consumption paths')
plt.show() 
#%% Extra plots II 4.2 - consumption policies
N_A_large = 200
A_GRID =  my_lin(-a_bar(r,1-sigma_y),a_max, N_A_large)
G_C1 = np.empty((N_A_large, N_y))
G_C2 = np.empty((N_A_large, N_y))
               
for i_l in range(0,N_y):
    g_c_func1 = interp1d(a_grid, cfunc_con1[:,i_l], fill_value="extrapolate",kind=method)
    g_c_func2 = interp1d(a_grid, cfunc_con2[:,i_l], fill_value="extrapolate",kind=method)

    G_C1[:,i_l]  = g_c_func1(A_GRID)
    G_C2[:,i_l]  = g_c_func2(A_GRID)
  
fig,ax = plt.subplots()
ax.plot(A_GRID, G_C1[:,0], label = 'Quadratic preferences and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(A_GRID, G_C2[:,0], label = 'CRRA preferences and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(A_GRID, G_C1[:,1], label = 'Quadratic preferences and income '+ str(y_grid[1]))
ax.plot(A_GRID, G_C2[:,1], label = 'CRRA preferences and income '+ str(y_grid[1]))
ax.set_xlabel('Assets today')
ax.legend()
ax.set_title('Consumption Policy $\sigma_y$ = {}'.format(sigma_y))
plt.show()    

  
#%% extra plot II4.2 - time paths Quadratic
y_1_in, a_1_in, c_1_in, v_1_in = path_sim2(0,y_grid[0],afunc_con1,vfunc_con1,N=2000)	
y_2_in, a_2_in, c_2_in, v_2_in = path_sim2(0,y_grid[1],afunc_con1,vfunc_con1,N=2000)	
y_3_in, a_3_in, c_3_in, v_3_in = path_sim2(2,y_grid[0],afunc_con1,vfunc_con1,N=2000)	
y_4_in, a_4_in, c_4_in, v_4_in = path_sim2(6,y_grid[0],afunc_con1,vfunc_con1,N=2000)	
y_5_in, a_5_in, c_5_in, v_5_in = path_sim2(15,y_grid[0],afunc_con1,vfunc_con1,N=2000)	

y_1_fin, a_1_fin, c_1_fin, v_1_fin = path_sim_fin2(0,y_grid[0],afunc_finite1,vfunc_finite1,N=2000)	
y_2_fin, a_2_fin, c_2_fin, v_2_fin = path_sim_fin2(0,y_grid[1],afunc_finite1,vfunc_finite1,N=2000)	
y_3_fin, a_3_fin, c_3_fin, v_3_fin = path_sim_fin2(2,y_grid[1],afunc_finite1,vfunc_finite1,N=2000)	
y_4_fin, a_4_fin, c_4_fin, v_4_fin = path_sim_fin2(6,y_grid[1],afunc_finite1,vfunc_finite1,N=2000)	
y_5_fin, a_5_fin, c_5_fin, v_5_fin = path_sim_fin2(15,y_grid[1],afunc_finite1,vfunc_finite1,N=2000)	

fig,ax = plt.subplots()
ax.plot(np.arange(T+1), c_1_in, label = 'assets=0 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
#ax.plot(np.arange(T+1), c_1_in, label = 'assets=0 finite horizon and income '+ str(y_grid[1]),linestyle='dashed')
ax.plot(np.arange(T+1), c_3_in, label = 'assets=2 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(np.arange(T+1), c_4_in, label = 'assets=6 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
#ax.plot(np.arange(T+1), c_5_in, label = 'assets=15 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(np.arange(T+1), c_1_fin, label = 'assets=0 infinite horizon and income '+ str(y_grid[0]))
#ax.plot(np.arange(T+1), c_2_fin, label = 'assets=0 infinite horizon and income '+ str(y_grid[1]))
ax.plot(np.arange(T+1), c_3_fin, label = 'assets=2 infinite horizon and income '+ str(y_grid[1]))
ax.plot(np.arange(T+1), c_4_fin, label = 'assets=6 infinite horizon and income '+ str(y_grid[1]))
#ax.plot(np.arange(T+1), c_5_fin, label = 'assets=15 infinite horizon and income '+ str(y_grid[0]))
ax.set_xlabel('time')
ax.legend()
ax.set_title('Consumption paths quadratic utilities $\sigma_y$= {} $\gamma$= {}'.format(sigma_y,gamma))
plt.show()  
 
#%% paths II 4.2 CRRA
y_1_in, a_1_in, c_1_in, v_1_in = path_sim2(0,y_grid[0],afunc_con2,vfunc_con2,N=2000)	
y_2_in, a_2_in, c_2_in, v_2_in = path_sim2(0,y_grid[1],afunc_con2,vfunc_con2,N=2000)	
y_3_in, a_3_in, c_3_in, v_3_in = path_sim2(2,y_grid[0],afunc_con2,vfunc_con2,N=2000)	
y_4_in, a_4_in, c_4_in, v_4_in = path_sim2(6,y_grid[0],afunc_con2,vfunc_con2,N=2000)	
y_5_in, a_5_in, c_5_in, v_5_in = path_sim2(15,y_grid[0],afunc_con2,vfunc_con2,N=2000)	

y_1_fin, a_1_fin, c_1_fin, v_1_fin = path_sim_fin2(0,y_grid[0],afunc_finite2,vfunc_finite2,N=2000)	
y_2_fin, a_2_fin, c_2_fin, v_2_fin = path_sim_fin2(0,y_grid[1],afunc_finite2,vfunc_finite2,N=2000)	
y_3_fin, a_3_fin, c_3_fin, v_3_fin = path_sim_fin2(2,y_grid[1],afunc_finite2,vfunc_finite2,N=2000)	
y_4_fin, a_4_fin, c_4_fin, v_4_fin = path_sim_fin2(6,y_grid[1],afunc_finite2,vfunc_finite2,N=2000)	
y_5_fin, a_5_fin, c_5_fin, v_5_fin = path_sim_fin2(15,y_grid[1],afunc_finite2,vfunc_finite2,N=2000)	

fig,ax = plt.subplots()
ax.plot(np.arange(T+1), c_1_in, label = 'assets=0 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
#ax.plot(np.arange(T+1), c_1_in, label = 'assets=0 finite horizon and income '+ str(y_grid[1]),linestyle='dashed')
ax.plot(np.arange(T+1), c_3_in, label = 'assets=2 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(np.arange(T+1), c_4_in, label = 'assets=6 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
#ax.plot(np.arange(T+1), c_5_in, label = 'assets=15 finite horizon and income '+ str(y_grid[0]),linestyle='dashed')
ax.plot(np.arange(T+1), c_1_fin, label = 'assets=0 infinite horizon and income '+ str(y_grid[0]))
#ax.plot(np.arange(T+1), c_2_fin, label = 'assets=0 infinite horizon and income '+ str(y_grid[1]))
ax.plot(np.arange(T+1), c_3_fin, label = 'assets=2 infinite horizon and income '+ str(y_grid[1]))
ax.plot(np.arange(T+1), c_4_fin, label = 'assets=6 infinite horizon and income '+ str(y_grid[1]))
#ax.plot(np.arange(T+1), c_5_fin, label = 'assets=15 infinite horizon and income '+ str(y_grid[0]))
ax.set_xlabel('time')
ax.legend()
ax.set_title('Consumption paths CRRA utility $\sigma_y$= {} $\gamma$= {}'.format(sigma_y,gamma))
plt.show()  
 