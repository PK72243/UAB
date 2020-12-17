import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from quantecon import Kalman, LinearStateSpace
from scipy.stats import uniform,gamma,norm,invgamma,beta,t
import seaborn as sns

#%% Solving for matrix coeficients (psi,phi) for my model (already reduced) using Blanchard-Kahn algorithm 
# The biggest part of code comes from 
# http://www.chadfulton.com/topics/estimating_rbc.html

reduced_equation_names = [
    'capital accumulation','euler equation'
]
reduced_variable_names = [
    'capital', 'consumption'
]
reduced_parameter_names = [
    'discount rate', 'risk aversion','depreciation rate','aggregate labour supply elasticity',
     'capital share',
    'technology shock persistence',
    'mean of domestic shock' ,'standard deviation of domestic shock',
    'mean of foreign shock' ,'standard deviation of foreign shock'
]

# Save some symbolic forms for pretty-printing
reduced_variable_symbols = [
    r"k", r"c"
]
reduced_contemporaneous_variable_symbols = [
    r"$%s_t$" % symbol for symbol in reduced_variable_symbols
]
reduced_lead_variable_symbols = [
    r"$%s_{t+1}$" % symbol for symbol in reduced_variable_symbols
]

reduced_parameter_symbols =  [
    r"$\beta$",r"$\theta$", r"$\delta$", r"$v$", r"$\alpha$", r"$\rho$",
    r"$\mu_{1}$", r"$\sigma1^2$", r"$\mu_{2}$", r"$\sigma2^2$"
]

class ReducedRBC1(object):
	def __init__(self, params=None):
		
		# Model dimensions
		self.k_params = 10
		self.k_variables = 2
		# Initialize parameters
		if params is not None:
			self.update(params)
		
		
	def update(self, params):
		# Save deep parameters
		self.discount_rate = params[0]
		self.risk_aversion = params[1]
		self.depreciation_rate = params[2]
		self.aggregate_labour_supply_elasticity = params[3]
		self.capital_share = params[4]
		self.technology_shock_persistence = params[5]
		self.domestic_mean = params[6]
		self.domestic_var = params[7]
		self.foreign_mean = params[8]
		self.foreign_var = params[9]
	
					
	def A(self):
		return np.eye(self.k_variables)
		
	def B(self): # In B and C I substitute the parameters from the reduced version of the model
		R_=self.capital_share*self.depreciation_rate/(1/self.discount_rate+self.depreciation_rate-1)
		c=(1-self.capital_share)/(self.capital_share+1/self.aggregate_labour_supply_elasticity)
		a=(1-(1-self.discount_rate*(1-self.depreciation_rate))*self.risk_aversion*c)**(-1)
		b=(1-self.discount_rate*(1-self.depreciation_rate))*(self.capital_share-1)*(1-self.capital_share/(1+1/self.aggregate_labour_supply_elasticity))
		B11 = 1 - self.depreciation_rate+self.capital_share*self.depreciation_rate/R_*(c+1)
		B12 = -self.depreciation_rate/R_*(c*self.risk_aversion+1+R_)
		B21=B11*b*a
		B22=a*(1+b*B12)
		return np.array([[B11, B12],[B21, B22]])
	def C(self):
		R_=self.capital_share*self.depreciation_rate/(1/self.discount_rate+self.depreciation_rate-1)
		c=(1-self.capital_share)/(self.capital_share+1/self.aggregate_labour_supply_elasticity)
		a=(1-(1-self.discount_rate*(1-self.depreciation_rate))*self.risk_aversion*c)**(-1)
		b=(1-self.discount_rate*(1-self.depreciation_rate))*(self.capital_share-1)*(1-self.capital_share/(1+1/self.aggregate_labour_supply_elasticity))
		C1 = self.depreciation_rate/R_*(c+1)
		C2 = a*b/(self.capital_share-1)*self.technology_shock_persistence+b*a*C1
		return np.array([C1, C2])[:,np.newaxis]
     

def ordered_jordan_decomposition(matrix):
    # Get eigenvalues and left eigenvectors of `matrix`
    # Note that the left eigenvectors of `matrix`
    # are the transpose of the right eigenvectors of the
    # transpose of matrix, and that `matrix` and `matrix'`
    # have the same eigenvalues
    eigenvalues, right_eigenvectors = np.linalg.eig(matrix.transpose())
    left_eigenvectors = right_eigenvectors.transpose()
    
    # Sort in increasing order
    idx = np.argsort(eigenvalues)
    
    # Return eigenvector matrix, diagonal eigenvalue matrix
    # Note that the left eigenvectors are in the rows now,
    # not the columns
    return np.diag(eigenvalues[idx]), left_eigenvectors[idx, :]

def solve_blanchard_kahn(B, C, rho, k_predetermined):
    # Perform the Jordan decomposition
    # this yields J, Lambda
    eigenvalues, left_eigenvectors = ordered_jordan_decomposition(B)
    left_eigenvectors = left_eigenvectors

    # Dimensions
    k_variables = len(B)
    k_nonpredetermined = k_variables - k_predetermined

    k_stable = len(np.where(eigenvalues.diagonal() < 1)[0])
    k_unstable = k_variables - k_stable

    # Make sure we're saddle-path stable
    if not k_unstable == k_nonpredetermined:
        raise RuntimeError('Blanchard-Kahn condition not met. Unique solution does not exist')

    # Decouple the system (this is D, above)
    decoupled_C = np.dot(left_eigenvectors, C)

    # Create partition indices
    p1 = np.s_[:k_predetermined]
    p2 = np.s_[k_predetermined:]

    p11 = np.s_[:k_predetermined, :k_predetermined]
    p12 = np.s_[:k_predetermined, k_predetermined:]
    p21 = np.s_[k_predetermined:, :k_predetermined]
    p22 = np.s_[k_predetermined:, k_predetermined:]

    # Solve the explosive component (controls) in terms of the
    # non-explosive component (states) and shocks
    tmp = np.linalg.inv(left_eigenvectors[p22])
    
    # This is \phi_{ck}, above
    policy_state = - np.dot(tmp, left_eigenvectors[p21])
    # This is \phi_{cz}, above
    policy_shock = -(
        np.dot(tmp, 1. / eigenvalues[p22]).dot(
            np.linalg.inv(
                np.eye(k_nonpredetermined) -
                rho / eigenvalues[p22]
            )
        ).dot(decoupled_C[p2])
    )

    # Solve for the non-explosive transition
    # This is T_{kk}, above
    transition_state = B[p11] + np.dot(B[p12], policy_state)
    # This is T_{kz}, above
    transition_shock = np.dot(B[p12], policy_shock) + C[p1]
    
    return policy_state, policy_shock, transition_state, transition_shock

class ReducedRBC2(ReducedRBC1):
    def solve(self, params=None):
        # Update the model parameters, if given
        if params is not None:
            self.update(params)
        
        # Solve the model
        phi_ck, phi_cz, T_kk, T_kz = solve_blanchard_kahn(
            self.B(), self.C(),
            self.technology_shock_persistence, 1
        )
        
               
        return  phi_ck, phi_cz, T_kk, T_kz

	    
#%% Other functions I use   
def random_coin(p):    # Drawing a coin, useful for MJ algorithm
    unif = np.random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
def hessian(x): #To spare the time I found this function on the web:
	#https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def coeff(params):# for a given parameters function compute matrices phi,psi
	
	discount_rate = params[0]
	risk_aversion = params[1]
	depreciation_rate = params[2]
	aggregate_labour_supply_elasticity = params[3]
	capital_share = params[4]
	technology_shock_persistence = params[5]
	domestic_mean = params[6]
	domestic_var = params[7]
	if domestic_var<=0:
		raise RuntimeError('Variance cennot be negative')

	foreign_mean = params[8]
	foreign_var = params[9]
	if foreign_var<=0:
		raise RuntimeError('Variance cennot be negative')

	parameters = pd.DataFrame({
    'name': reduced_parameter_names,
    'value': [discount_rate, risk_aversion,depreciation_rate,aggregate_labour_supply_elasticity, capital_share,technology_shock_persistence
	      ,domestic_mean,domestic_var,foreign_mean,foreign_var]})  
	reduced_mod = ReducedRBC2(parameters['value'])

	# Check the Blanchard-Kahn conditions
	eigenvalues, left_eigenvectors = ordered_jordan_decomposition(reduced_mod.B())
	#print('Modulus of eigenvalues of B:', np.abs(eigenvalues.diagonal()))
	c1,c2,c3,c4=reduced_mod.solve()
	phi1=np.array([c3,c4,0,0,reduced_mod.technology_shock_persistence,domestic_mean+foreign_mean,0,0,1],dtype='float64').reshape(3,3)
	phi2=np.array([0,np.sqrt(domestic_var)+c4*foreign_var,0],dtype='float64').reshape(3,1)
	psi=np.array([c1,c2,0],dtype='float64').reshape(1,3)
	return phi1,phi2,psi

def likelihood_params(phi1,phi2,psi,data,sigma_v):
	# For the given matrices the moments for the likelihood function are obtained using a Kalman filter
	ss=LinearStateSpace(phi1,phi2,psi,np.ones((1,1))*np.sqrt(sigma_v))	
	kalman=Kalman(ss,[0,0,1],[[1000,1000,1000],[1000,1000,1000],[1000,1000,1000]])
	mu=[]
	sigma=[]
	for i in range(len(data)):
		kalman.update(data[i])
		mu.append(psi.dot(kalman.x_hat))
		sigma.append(psi.dot(kalman.Sigma).dot(np.transpose(psi))+sigma_v**2)
	
	return mu[-1],sigma[-1]

def joint_prob(params):
	# For the given parameters I compute the joint denisty
	# Here I store the priors!!!!!!!
	betas = params[0]
	theta = params[1]
	delta = params[2]
	v = params[3]
	alpha = params[4]
	rho = params[5]
	mu_d=params[6]
	sigma_d = params[7]
	mu_f=params[8]
	sigma_f = params[9]
	
	p1=beta.pdf(betas,9.28,0.386)
	p2=gamma.pdf(theta,2)
	p3=beta.pdf(delta,89.9,809.1)
	p4=gamma.pdf(v,2)
	p5=beta.pdf(alpha,182,369.5)
	p6=beta.pdf(rho,162.88,6.78)
	p7=norm.pdf(mu_d,0,0.04)
	p8=invgamma.pdf(sigma_d,2.0025,0,0.010025)
	p9=norm.pdf(mu_f,0,0.04)
	p10=invgamma.pdf(sigma_f,2.0625,0,0.053125)
	
	P=p1*p2*p3*p4*p5*p6*p7*p8*p9*p10
	return P

def likelihood(mu,sigma,data):
	#likelihood function for given mean, variance and data
	ll=1
	for i in range(len(data)):
		ll=ll*norm.pdf(data[i],mu,sigma)
		#print(norm.pdf(data[i],mu,sigma))
	return ll
#%% Some parameters
c=0.0004 #parameter used for getting the variance of the drawing posterior distribution
sigma_v=0.2 #standard deviation of measurement error
c_dev=np.load('c.npy')[-30:] #loading log deviations of consumption


#%% Drawing conditional distribution 
# I wrote the function that should be minimze to obtain the drawing distribution for the random walk MH algorithm
# I had some serious problem with obtaining satisfactory results,
# hence I use below very brute force method, I also need to obtain inverse of a negative hessian
def dist_draw(params):
	try:
		discount_rate = params[0]
		risk_aversion = params[1]
		depreciation_rate = params[2]
		aggregate_labour_supply_elasticity = params[3]
		capital_share = params[4]
		technology_shock_persistence = params[5]
		domestic_mean = params[6]
		domestic_var = params[7]
		foreign_mean = params[8]
		foreign_var = params[9]
		
		if domestic_var<=0:
			return 99999
		elif foreign_var<=0:
			return 99999
		else:
			phi1,phi2,psi=coeff(params)
			a,b=likelihood_params(phi1,phi2,psi,c_dev,sigma_v)
			likelihood(a,b,c_dev)
			if joint_prob(params)==0:
				return 99999
			else:
				return -np.log(likelihood(a,b,c_dev))-np.log(joint_prob(params))
	except RuntimeError :
		return 99999
	except ValueError:
		return 99999
	except IndexError:
		return 99999
#%% Minimizing it with in this way procduces some unproper results
dd=minimize(dist_draw,[0.98,2,0.04,2,0.33,0.9,0,1,0,1])
mu_draw=dd.x
sigma_draw=dd.hess_inv*c**2
#%%#%% Second method 
# That is why I use some simple brute force solver
# I just discretize the space around means of distribution		
mu_draw=np.array((0.96,2,0.1,2,0.33,0.96,0,0.1,0,0.5)).reshape(10)
var=np.array((0.004,1,0.01**2,1,0.004,0.015**2,0.04,4,0.04,4))
#%%
dim=3
beta_g=np.linspace(mu_draw[0]-0.2*var[0],mu_draw[0]+0.2*var[0],dim)
theta_g=np.linspace(mu_draw[1]-0.2*var[1],mu_draw[1]+0.2*var[1],dim)
delta_g=np.linspace(mu_draw[2]-0.2*var[2],mu_draw[2]+0.2*var[2],dim)
v_g=np.linspace(mu_draw[3]-0.2*var[3],mu_draw[3]+0.2*var[3],dim)
alpha_g=np.linspace(mu_draw[4]-0.2*var[4],mu_draw[4]+0.2*var[4],dim)
rho_g=np.linspace(mu_draw[5]-0.2*var[5],mu_draw[5]+0.2*var[5],dim)
mu_1_g=np.linspace(mu_draw[6]-0.2*var[6],mu_draw[6]+0.2*var[6],dim)
std_1_g=np.linspace(mu_draw[7]-0.02*var[7],mu_draw[7]+0.02*var[7],dim)
mu_2_g=np.linspace(mu_draw[8]-0.2*var[8],mu_draw[8]+0.2*var[8],dim)
std_2_g=np.linspace(mu_draw[9]-0.02*var[9],mu_draw[9]+0.02*var[9],dim)

#%% Second method vol.2 it takes 15-20 minutes
R=np.array(np.zeros((dim,dim,dim,dim,dim,dim,dim,dim,dim,dim)))
for i_a in range(dim):
	print(i_a)
	for i_b in range(dim):
		print(i_b)
		for i_c in range(dim):
			for i_d in range(dim):
				for i_e in range(dim):
					for i_f in range(dim):
						for i_g in range(dim):
							for i_h in range(dim):
								for i_i in range(dim):
									for i_j in range(dim):
										try:
											params=[beta_g[i_a],theta_g[i_b],delta_g[i_c],v_g[i_d],alpha_g[i_e],rho_g[i_f],mu_1_g[i_g],std_1_g[i_h],mu_2_g[i_i],std_2_g[i_j]]
											phi1,phi2,psi=coeff(params)
											mu,sig=likelihood_params(phi1,phi2,psi,c_dev,sigma_v)
											if likelihood(mu,sig,c_dev)==0:
												R[i_a,i_b,i_c,i_d,i_e,i_f,i_g,i_h,i_i,i_j]=999999999
											else:
												R[i_a,i_b,i_c,i_d,i_e,i_f,i_g,i_h,i_i,i_j]=-np.log(likelihood(mu,sig,c_dev))-np.log(joint_prob(params))
										except:
											R[i_a,i_b,i_c,i_d,i_e,i_f,i_g,i_h,i_i,i_j]=999999

np.save('R',R)
#%%
#np.load('R.npy')
np.amin(R)
aa,ab,ac,ad,ae,af,ag,ah,ai,aj=np.where(R == np.amin(R))	
hess=hessian(R)[:,:,aa,ab,ac,ad,ae,af,ag,ah,ai,aj].reshape(10,10)
mu_draw=np.array((beta_g[aa],theta_g[ab],delta_g[ac],v_g[ad],alpha_g[ae],rho_g[af],mu_1_g[ag],std_1_g[ah],mu_2_g[ai],std_2_g[aj])).reshape(10)
sigma_draw=np.linalg.inv(hess)*(-1)
sigma_draw=np.sign(sigma_draw)*np.log(np.abs(sigma_draw))*c**2
		

#%% Random Walk MH algorithm
def mcmc(mu_draw,sigma_draw,N=10,burn_in=0.2):
	states = [] #storing results here
	icc=0
	burn_in = int(N*burn_in) #discarding first results
	phi0=None # I use this conditions (while x is None) to avoid situation in which a set of parameters
	# that does not satisfy Blanchard_Kahn conditions or with the negative variance  is drawn
	while phi0 is None:
		try:
			current=np.random.multivariate_normal(mu_draw, sigma_draw, 1) #drawing the first set of parameters
			phi0,phi0_,psi_=coeff(current[-1]) #checking if is a proper one
		except:
			pass
	for i in range(N):

		states.append(current)
		phi1,phi2,psi=coeff(current[-1])
		a,b=likelihood_params(phi1,phi2,psi,c_dev,sigma_v)
		curr_prob = likelihood(a,b,c_dev)*joint_prob(current[-1]) #computing the posterior probability
		
		phi3=None
		while phi3 is None:
			try:
				movement=np.random.multivariate_normal(current[0], sigma_draw, 1) #drawing next set
				phi3,phi4,psi2=coeff(movement[0])
				e,d=likelihood_params(phi3,phi4,psi2,c_dev,sigma_v) #computing the posterior probability
			except:
				continue
		
		move_prob = likelihood(e,d,c_dev)*joint_prob(movement[0]) 
		acceptance = min(move_prob/curr_prob,1) #Assesing the jump probability
		if random_coin(acceptance):
			current = movement
			icc=icc+1
		if i % 100 ==0:
			print(i)
	return np.array(states[burn_in:]).reshape(N-burn_in,10),icc/N
	

#%% For 1 million draws it takes about 3 hours
# There are some warings for determinant of covariance matrix, but it is still positive and very small
tst,acc=mcmc(mu_draw,sigma_draw,N=1000000)
#%% Saving		
np.save('mcmc',tst)	
np.save('acc',acc)	
#%%Loading
mcmc=np.load('mcmc.npy')
				
#%%Graphs
x_6=np.empty((1000))
for i in range(1000):
	x_6[i]=np.random.normal(0,0.04,1)
#Plot of domestic mean density	
fig, ax = plt.subplots()
sns.distplot(tst[:,6], label='Posterior distribution')
sns.distplot(x_6, label='Prior distribution',hist=False)
plt.title('Distribution of $\mu_d$')
plt.ylabel("Density")
plt.legend()
plt.show()	
# Plot of foreign mean density
fig, ax = plt.subplots()
sns.distplot(tst[:,8], label='Posterior distribution')
sns.distplot(x_6, label='Prior distribution',hist=False)
plt.title('Distribution of $\mu_f$')
plt.ylabel("Density")
plt.legend()
plt.show()

for i in range(len(tst[0])):
	mu=np.mean(tst[:,i])
	var=np.var(tst[:,i])
	t_v=mu/np.sqrt(var/len(tst[:,i]))
	ttest=t.cdf(t_v,len(tst[:,i])-1)
	print(i,mu,var,ttest)
		

