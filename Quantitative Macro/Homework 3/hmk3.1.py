import numpy as np
import time
import matplotlib.pyplot as plt

#parameters
kappa=5.24
v=2
theta=0.679
delta=0.013
beta=0.988
epsilon=0.1
#grid parameters
beginning_point=0.2
end_point=60
gridsize=300

#Utility function
def u(c,h=1,kappa=kappa,v=v,end_labour=False):
	if (end_labour==True):
		return np.log(c)-kappa*((h**(1+1/v))/(1+1/v))
	else:
		return np.log(c)
#Components of M matrix
def m(ki,kj,theta=theta,delta=delta):
	return u(ki**(1-theta)-kj+(1-delta)*ki)

###for the brute force VFI I provide a detailed comments of the code, in the following methods I will only comment on the code changes

#generating a grid
k=np.linspace(beginning_point,end_point,gridsize)
#Creating a M matrix
M=np.zeros((gridsize,gridsize),dtype='float64')
for i,k_i in enumerate(k):
		for j,k_j in enumerate(k):
			if(k_i**(1-theta)-k_j+(1-delta)*k_i<=0):	# if the consumption is negative then I attach -999999
				M[i,j]=-999999
			else:
				M[i,j]=m(k_i,k_j)

start_time=time.time() #time stopper
X=np.zeros((gridsize,gridsize),dtype='float64')	 #initialize a matrix of X with zeros everywhere
V=[np.zeros((gridsize,1),dtype='float64')]	#initial guess for the value function, as you can notice I decided to store value of each iteration for deeeper
# analysis, that is the reason why I create here and below a list of lists
gk=[np.zeros((gridsize,1),dtype='int')]	#initialze policy function for k, here I store indexes of k not values, because than the algorithm in my case is faster
gc=[np.zeros((gridsize,1),dtype='float64')] #initialze a policy function for c
distance=np.array([epsilon+1],dtype='float64') #initialze a distance with 1+epsilon, so it passes through the first loop condition

counter=0	#counter for iteration
while(distance[counter]>epsilon):	#loop for iterating
	V.append(np.zeros((gridsize,1),dtype='float64'))	#initialze value function in the next period
	gk.append(np.zeros((gridsize,1),dtype='int'))		#initialize policy function for capital in the next period
	gc.append(np.zeros((gridsize,1),dtype='float64'))	#initialze policy function for consumption in the next period
	for i in range(gridsize):
		for j in range(gridsize):
			X[i,j]=M[i,j]+beta*V[counter][j]	#X matrix is calculated
		V[counter+1][i]=max(X[i,:])		#value function in the next period
		gk[counter+1][i]=np.argmax(X[i,:])	#policy function in the next period
		gc[counter+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk[counter+1][i]] 
	distance=np.append(distance,[(sum(np.subtract(V[counter],V[counter+1])**(2)))**(0.5)]) #distance between two value functions
	counter+=1
VFI_time=time.time()-start_time	#end of timing

#VFI comparison
fig,ax=plt.subplots()
ax.plot(k,V[-1],color='black',label="{} - last one, took {} seconds".format(counter,round(VFI_time,2)),linewidth=2.5)
ax.plot(k,V[1],color='green',label="iteration 1")
ax.plot(k,V[100],color='blue',label="iteration 100")
ax.plot(k,V[200],color='brown',label="iteration 200")
ax.set_title("Value function after iterations")
ax.set_ylabel("Value Fubction")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
#Policy functions
fig, axs = plt.subplots(1,2,figsize=(21,8))
axs[0].plot(k,k[gk[-1]])
axs[0].set_title("Policy function for $k_{t+1}$",size=22)
axs[0].set_ylabel("$k_{t+1}$",size=16)
axs[0].set_xlabel("$k_{t}$",size=16)
axs[1].plot(k,gc[-1])
axs[1].set_title("Policy function for ct",size=22)
axs[1].set_ylabel("$c_{t}$",size=16)
axs[1].set_xlabel("$k_{t}$",size=16)
plt.show()
#distance
fig,ax=plt.subplots()
ax.plot(distance[1:])
ax.set_title("Distance function")
ax.set_xlabel("Iterations")
ax.axhline(epsilon,color='black',lw=1,ls='--')
plt.show()

####################################
####### MONOTONICITY ##############
start_time=time.time()
X_mon=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_mon=[np.zeros((gridsize,1),dtype='float64')]
gk_mon=[np.zeros((gridsize,1),dtype='int')]
gc_mon=[np.zeros((gridsize,1),dtype='float64')]
distance_mon=np.array([epsilon+1],dtype='float64')
counter_mon=0
while(distance_mon[counter_mon]>epsilon):
	V_mon.append(np.zeros((gridsize,1),dtype='float64'))
	gk_mon.append(np.zeros((gridsize,1),dtype='int'))
	gc_mon.append(np.zeros((gridsize,1),dtype='float64'))
	for i in range(gridsize):
		if (i==0):						# let us compute the first row first
			for j in range(gridsize):
				X[i,j]=M[i,j]+beta*V[counter][j]
		else:
			g=gk_mon[counter_mon+1][i-1] #optimal policy for kapital in i-th row
			for j in range(gridsize):
				if(j>=g):	# we only look for columns on the right of this optimal capital
					X_mon[i,j]=M[i,j]+beta*V_mon[counter_mon][j]
				else:
					continue
		V_mon[counter_mon+1][i]=max(X_mon[i,:])
		gk_mon[counter_mon+1][i]=np.argmax(X_mon[i,:])
		gc_mon[counter_mon+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[int(gk_mon[counter_mon+1][i])]
	distance_mon=np.append(distance_mon,[(sum(np.subtract(V[counter_mon],V[counter_mon+1])**(2)))**(0.5)])
	counter_mon+=1
VFI_time_mon=time.time()-start_time	

#VFI plot
fig,ax=plt.subplots()
ax.plot(k,V_mon[-1],color='black',label="{} iterations, took {} seconds".format(counter_mon,round(VFI_time_mon,2)),linewidth=2.5)
ax.set_title("Value function with monotonicity")
ax.set_ylabel("Value Function")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
fig,ax=plt.subplots()
ax.plot(k,gk_mon[-1],color='black',label="{} iterations, took {} seconds".format(counter_mon,round(VFI_time_mon,2)),linewidth=2.5)
ax.set_title("Value function with monotonicity")
ax.set_ylabel("$k_{t+1}$")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

####################################
####### CONCAVITY ##############
start_time=time.time()
X_con=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_con=[np.zeros((gridsize,1),dtype='float64')]
gk_con=[np.zeros((gridsize,1),dtype='int')]
gc_con=[np.zeros((gridsize,1),dtype='float64')]
distance_con=np.array([epsilon+1],dtype='float64')
counter_con=0
while(distance_con[counter_con]>epsilon):
	
	V_con.append(np.zeros((gridsize,1),dtype='float64'))
	gk_con.append(np.zeros((gridsize,1),dtype='int'))
	gc_con.append(np.zeros((gridsize,1),dtype='float64'))
	for i in range(gridsize):
		for j in range(gridsize):
			X_con[i,j]=M[i,j]+beta*V_con[counter_con][j]
			if (j>0 and X_con[i,j-1]>X_con[i,j]):		#if values in row start to decrease we break the llop
				break
		V_con[counter_con+1][i]=X_con[i,j-1]		
		gk_con[counter_con+1][i]=j-1
		gc_con[counter_con+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[int(gk_con[counter_con+1][i])]
				
	distance_con=np.append(distance_con,[(sum(np.subtract(V[counter_con],V[counter_con+1])**(2)))**(0.5)])
	counter_con+=1
VFI_time_con=time.time()-start_time	

#VFI plot
fig,ax=plt.subplots()
ax.plot(k,V_con[-1],color='black',label="{} iterations, took {} seconds".format(counter_con,round(VFI_time_con,2)),linewidth=2.5)
ax.set_title("Value function with concavity")
ax.set_ylabel("Value Fubction")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

fig,ax=plt.subplots()
ax.plot(k,gk_con[-1],color='black',label="{} iterations, took {} seconds".format(counter_mon,round(VFI_time_mon,2)),linewidth=2.5)
ax.set_title("Value function with concavity")
ax.set_ylabel("$k_{t+1}$")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

####################################
####### CONCAVITY + MONOTONICITY ##############
start_time=time.time()
X_cmon=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_cmon=[np.zeros((gridsize,1),dtype='float64')]
gk_cmon=[np.zeros((gridsize,1),dtype='int')]
gc_cmon=[np.zeros((gridsize,1),dtype='float64')]
distance_cmon=np.array([epsilon+1],dtype='float64')
counter_cmon=0
while(distance_cmon[counter_cmon]>epsilon):
	
	V_cmon.append(np.zeros((gridsize,1),dtype='float64'))
	gk_cmon.append(np.zeros((gridsize,1),dtype='int'))
	gc_cmon.append(np.zeros((gridsize,1),dtype='float64'))
	for i in range(gridsize):
		if (i==0):						#monotonicity
			for j in range(gridsize):
				X[i,j]=M[i,j]+beta*V[counter][j]
		else:
			g=gk_cmon[counter_cmon][i]		#monotonicity
			for j in range(gridsize):
				if(j>=g):					#monotonicity
					X_cmon[i,j]=M[i,j]+beta*V_cmon[counter_cmon][j]
					if (j>0 and X_cmon[i,j-1]>X_cmon[i,j]):		#concavity
						break
					else:
						continue
		V_cmon[counter_cmon+1][i]=X_cmon[i,j-1]		
		gk_cmon[counter_cmon+1][i]=j-1
		gc_cmon[counter_cmon+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[int(gk_cmon[counter_cmon+1][i])]
				
	distance_cmon=np.append(distance_cmon,[(sum(np.subtract(V[counter_cmon],V[counter_cmon+1])**(2)))**(0.5)])
	counter_cmon+=1
VFI_time_cmon=time.time()-start_time	

#VFI plot
fig,ax=plt.subplots()
ax.plot(k,V_cmon[-1],color='black',label="{} iterations, took {} seconds".format(counter_cmon,round(VFI_time_cmon,2)),linewidth=2.5)
ax.set_title("Value function with concavity + monotonicity")
ax.set_ylabel("Value Fubction")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

fig,ax=plt.subplots()
ax.plot(k,gk_cmon[-1],color='black',label="{} iterations, took {} seconds".format(counter_mon,round(VFI_time_mon,2)),linewidth=2.5)
ax.set_title("Value function with concavity + monotonicity")
ax.set_ylabel("$k_{t+1}$")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
####################################
####### LOCAL SEARCH ##############
start_time=time.time()
X_loc=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_loc=[np.zeros((gridsize,1),dtype='float64')]
gk_loc=[np.zeros((gridsize,1),dtype='int')]
gc_loc=[np.zeros((gridsize,1),dtype='float64')]
distance_loc=np.array([epsilon+1],dtype='float64')
counter_loc=0
while(distance[counter_loc]>epsilon):
	V_loc.append(np.zeros((gridsize,1),dtype='float64'))
	gk_loc.append(np.zeros((gridsize,1),dtype='int'))
	gc_loc.append(np.zeros((gridsize,1),dtype='float64'))
	for i in range(gridsize):
		for j in range(gridsize):
			if (i==0):
				X_loc[i,j]=M[i,j]+beta*V_loc[counter_loc][j]		# first, I compute the whole first row
			elif(j==gk_loc[counter_loc+1][i-1]):					#if kj is the optimal capital of the previous row, the solution for the current row should be near
				for m in range(j-10,j+10):
					if(m>=0 and m<gridsize):
						X_loc[i,m]=M[i,m]+beta*V_loc[counter_loc][m]
					else:
						continue
			else:
				continue
		V_loc[counter_loc+1][i]=max(X_loc[i,:])
		gk_loc[counter_loc+1][i]=np.argmax(X_loc[i,:])
		gc_loc[counter_loc+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk[counter_loc+1][i]]
				
	distance_loc=np.append(distance_loc,[(sum(np.subtract(V[counter_loc],V[counter_loc+1])**(2)))**(0.5)])
	counter_loc+=1
VFI_time_loc=time.time()-start_time	

#VFI plot
fig,ax=plt.subplots()
ax.plot(k,V_loc[-1],color='black',label="{} iterations, took {} seconds".format(counter_loc,round(VFI_time_loc,2)),linewidth=2.5)
ax.set_title("Value function with local search")
ax.set_ylabel("Value Fubction")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

fig,ax=plt.subplots()
ax.plot(k,gk_mon[-1],color='black',label="{} iterations, took {} seconds".format(counter_mon,round(VFI_time_mon,2)),linewidth=2.5)
ax.set_title("Value function with local search")
ax.set_ylabel("$k_{t+1}$")
ax.set_xlabel("$k_{t}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
####################################
####### HOWARD ##############
## Iteration 1
start_time=time.time()
gk_guess=np.reshape((np.arange(0,300,1))+1,(gridsize,1))	#initial guess
gk_guess[299]=299
t_h1=1	# iteration where we apply the guess
X_how1=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how1=[np.zeros((gridsize,1),dtype='float64')]
gk_how1=[np.zeros((gridsize,1),dtype='int')]
gc_how1=[np.zeros((gridsize,1),dtype='float64')]
distance_how1=np.array([epsilon+1],dtype='float64')
counter_how1=0
while(distance_how1[counter_how1]>epsilon):
	V_how1.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how1.append(np.zeros((gridsize,1),dtype='int'))
	gc_how1.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how1+1==t_h1):
		gk_how1[counter_how1+1]=gk_guess 		#applying the guess
		for i in range(gridsize):
			gc_how1[counter_how1+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how1[counter_how1+1][i]]
			V_how1[counter_how1+1][i]=M[i,gk_guess[i]]+beta*V_how1[counter_how1][gk_guess[i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how1[i,j]=M[i,j]+beta*V_how1[counter_how1][j]
			V_how1[counter_how1+1][i]=max(X_how1[i,:])
			gk_how1[counter_how1+1][i]=np.argmax(X_how1[i,:])
			gc_how1[counter_how1+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how1[counter_how1+1][i]]			
	
	distance_how1=np.append(distance_how1,[(sum(np.subtract(V_how1[counter_how1],V_how1[counter_how1+1])**(2)))**(0.5)])
	counter_how1+=1
VFI_time_how1=time.time()-start_time	

#Iteration 50
start_time=time.time()
gk_guess=np.reshape((np.arange(0,300,1)),(gridsize,1))
t_h2=50
X_how2=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how2=[np.zeros((gridsize,1),dtype='float64')]
gk_how2=[np.zeros((gridsize,1),dtype='int')]
gc_how2=[np.zeros((gridsize,1),dtype='float64')]
distance_how2=np.array([epsilon+1],dtype='float64')
counter_how2=0
while(distance_how2[counter_how2]>epsilon):
	V_how2.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how2.append(np.zeros((gridsize,1),dtype='int'))
	gc_how2.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how2+1==t_h2):
		gk_how2[counter_how2+1]=gk_guess
		for i in range(gridsize):
			gc_how2[counter_how2+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how2[counter_how2+1][i]]
			V_how2[counter_how2+1][i]=M[i,gk_guess[i]]+beta*V_how2[counter_how2][gk_guess[i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how2[i,j]=M[i,j]+beta*V_how2[counter_how2][j]
			V_how2[counter_how2+1][i]=max(X_how2[i,:])
			gk_how2[counter_how2+1][i]=np.argmax(X_how2[i,:])
			gc_how2[counter_how2+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how2[counter_how2+1][i]]			
	distance_how2=np.append(distance_how2,[(sum(np.subtract(V_how2[counter_how2],V_how2[counter_how2+1])**(2)))**(0.5)])
	counter_how2+=1
VFI_time_how2=time.time()-start_time	

#Iteration 100
start_time=time.time()
gk_guess=np.reshape((np.arange(0,300,1)),(gridsize,1))
t_h3=100
X_how3=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how3=[np.zeros((gridsize,1),dtype='float64')]
gk_how3=[np.zeros((gridsize,1),dtype='int')]
gc_how3=[np.zeros((gridsize,1),dtype='float64')]
distance_how3=np.array([epsilon+1],dtype='float64')
counter_how3=0
while(distance_how3[counter_how3]>epsilon):
	V_how3.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how3.append(np.zeros((gridsize,1),dtype='int'))
	gc_how3.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how3+1==t_h3):
		gk_how3[counter_how3+1]=gk_guess
		for i in range(gridsize):
			gc_how3[counter_how3+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how3[counter_how3+1][i]]
			V_how3[counter_how3+1][i]=M[i,gk_guess[i]]+beta*V_how3[counter_how3][gk_guess[i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how3[i,j]=M[i,j]+beta*V_how3[counter_how3][j]
			V_how3[counter_how3+1][i]=max(X_how3[i,:])
			gk_how3[counter_how3+1][i]=np.argmax(X_how3[i,:])
			gc_how3[counter_how3+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how3[counter_how3+1][i]]			
	distance_how3=np.append(distance_how3,[(sum(np.subtract(V_how3[counter_how3],V_how3[counter_how3+1])**(2)))**(0.5)])
	counter_how3+=1
VFI_time_how3=time.time()-start_time	

##VFI comparison

fig,ax=plt.subplots()
ax.plot(V[-1]-V_how1[-1],color='blue',label="Howard startting at 1: {} iterations, took {} seconds".format(counter_how1,round(VFI_time_how1,2)))
ax.plot(V[-1]-V_how2[-1],color='yellow',label="Howard startting at 50: {} iterations, took {} seconds".format(counter_how2,round(VFI_time_how2,2)))
ax.plot(V[-1]-V_how3[-1],color='gray',label="Howard startting at 100: {} iterations, took {} seconds".format(counter_how3,round(VFI_time_how3,2)))
ax.set_title("Distance between functions")
ax.set_xlabel("number of iterations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

#########################################
####### Howard with reassesments ########
###5
start_time=time.time()
gk_guess=[np.reshape((np.arange(0,300,1))+1,(gridsize,1))]
gk_guess[0][299]=299
freq1=5		#frequency of reassesing
for i in range(counter//freq1):
	gk_guess.append(gk[freq1*(i+1)])		#guesses from the initial solution
X_how11=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how11=[np.zeros((gridsize,1),dtype='float64')]
gk_how11=[np.zeros((gridsize,1),dtype='int')]
gc_how11=[np.zeros((gridsize,1),dtype='float64')]
distance_how11=np.array([epsilon+1],dtype='float64')
counter_how11=0
while(distance_how11[counter_how11]>epsilon):
	V_how11.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how11.append(np.zeros((gridsize,1),dtype='int'))
	gc_how11.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how11%freq1==0):		#here guesses are implemented
		gk_how11[counter_how11+1]=gk_guess[counter_how11//freq1]
		for i in range(gridsize):
			gc_how11[counter_how11+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how11[counter_how11+1][i]]
			V_how11[counter_how11+1][i]=M[i,gk_guess[counter_how11//freq1][i]]+beta*V_how11[counter_how11][gk_guess[counter_how11//freq1][i]]
	else:								# standard procedure for non-guesses
		for i in range(gridsize):
			for j in range(gridsize):
				X_how11[i,j]=M[i,j]+beta*V_how11[counter_how11][j]
			V_how11[counter_how11+1][i]=max(X_how11[i,:])
			gk_how11[counter_how11+1][i]=np.argmax(X_how11[i,:])
			gc_how11[counter_how11+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how11[counter_how11+1][i]]			
	distance_how11=np.append(distance_how11,[(sum(np.subtract(V_how11[counter_how11],V_how11[counter_how11+1])**(2)))**(0.5)])
	counter_how11+=1
VFI_time_how11=time.time()-start_time
####10
start_time=time.time()
gk_guess=[np.reshape((np.arange(0,300,1))+1,(gridsize,1))]
gk_guess[0][299]=299
freq2=10
for i in range(counter//freq2):
	gk_guess.append(gk[freq2*(i+1)])
X_how12=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how12=[np.zeros((gridsize,1),dtype='float64')]
gk_how12=[np.zeros((gridsize,1),dtype='int')]
gc_how12=[np.zeros((gridsize,1),dtype='float64')]
distance_how12=np.array([epsilon+1],dtype='float64')
counter_how12=0
while(distance_how12[counter_how12]>epsilon):
	V_how12.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how12.append(np.zeros((gridsize,1),dtype='int'))
	gc_how12.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how12%freq2==0):
		gk_how12[counter_how12+1]=gk_guess[counter_how12//freq2]
		for i in range(gridsize):
			gc_how12[counter_how12+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how12[counter_how12+1][i]]
			V_how12[counter_how12+1][i]=M[i,gk_guess[counter_how12//freq2][i]]+beta*V_how12[counter_how12][gk_guess[counter_how12//freq2][i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how12[i,j]=M[i,j]+beta*V_how12[counter_how12][j]
			V_how12[counter_how12+1][i]=max(X_how12[i,:])
			gk_how12[counter_how12+1][i]=np.argmax(X_how12[i,:])
			gc_how12[counter_how12+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how12[counter_how12+1][i]]			
	distance_how12=np.append(distance_how12,[(sum(np.subtract(V_how12[counter_how12],V_how12[counter_how12+1])**(2)))**(0.5)])
	counter_how12+=1
VFI_time_how12=time.time()-start_time
####20
start_time=time.time()
gk_guess=[np.reshape((np.arange(0,300,1))+1,(gridsize,1))]
gk_guess[0][299]=299
freq3=20
for i in range(counter//freq3):
	gk_guess.append(gk[freq3*(i+1)])
X_how13=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how13=[np.zeros((gridsize,1),dtype='float64')]
gk_how13=[np.zeros((gridsize,1),dtype='int')]
gc_how13=[np.zeros((gridsize,1),dtype='float64')]
distance_how13=np.array([epsilon+1],dtype='float64')
counter_how13=0
while(distance_how13[counter_how13]>epsilon):
	V_how13.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how13.append(np.zeros((gridsize,1),dtype='int'))
	gc_how13.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how13%freq3==0):
		gk_how13[counter_how13+1]=gk_guess[counter_how13//freq3]
		for i in range(gridsize):
			gc_how13[counter_how13+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how13[counter_how13+1][i]]
			V_how13[counter_how13+1][i]=M[i,gk_guess[counter_how13//freq3][i]]+beta*V_how13[counter_how13][gk_guess[counter_how13//freq3][i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how13[i,j]=M[i,j]+beta*V_how13[counter_how13][j]
			V_how13[counter_how13+1][i]=max(X_how13[i,:])
			gk_how13[counter_how13+1][i]=np.argmax(X_how13[i,:])
			gc_how13[counter_how13+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how13[counter_how13+1][i]]			
	distance_how13=np.append(distance_how13,[(sum(np.subtract(V_how13[counter_how13],V_how13[counter_how13+1])**(2)))**(0.5)])
	counter_how13+=1
VFI_time_how13=time.time()-start_time
####50
start_time=time.time()
gk_guess=[np.reshape((np.arange(0,300,1))+1,(gridsize,1))]
gk_guess[0][299]=299
freq4=50
for i in range(counter//freq4):
	gk_guess.append(gk[freq4*(i+1)])
X_how14=np.full((gridsize,gridsize),-999999,dtype='float64')	
V_how14=[np.zeros((gridsize,1),dtype='float64')]
gk_how14=[np.zeros((gridsize,1),dtype='int')]
gc_how14=[np.zeros((gridsize,1),dtype='float64')]
distance_how14=np.array([epsilon+1],dtype='float64')
counter_how14=0
while(distance_how14[counter_how14]>epsilon):
	V_how14.append(np.zeros((gridsize,1),dtype='float64'))
	gk_how14.append(np.zeros((gridsize,1),dtype='int'))
	gc_how14.append(np.zeros((gridsize,1),dtype='float64'))
	if (counter_how14%freq4==0):
		gk_how14[counter_how14+1]=gk_guess[counter_how14//freq4]
		for i in range(gridsize):
			gc_how14[counter_how14+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how14[counter_how14+1][i]]
			V_how14[counter_how14+1][i]=M[i,gk_guess[counter_how14//freq4][i]]+beta*V_how14[counter_how14][gk_guess[counter_how14//freq4][i]]
	else:
		for i in range(gridsize):
			for j in range(gridsize):
				X_how14[i,j]=M[i,j]+beta*V_how14[counter_how14][j]
			V_how14[counter_how14+1][i]=max(X_how14[i,:])
			gk_how14[counter_how14+1][i]=np.argmax(X_how14[i,:])
			gc_how14[counter_how14+1][i]=k[i]**(1-theta)+(1-delta)*k[i]-k[gk_how14[counter_how14+1][i]]			
	distance_how14=np.append(distance_how14,[(sum(np.subtract(V_how14[counter_how14],V_how14[counter_how14+1])**(2)))**(0.5)])
	counter_how14+=1
VFI_time_how14=time.time()-start_time

##VFI comparison

fig,ax=plt.subplots()
ax.plot(V[-1]-V_how11[-1],color='blue',label="Howard's with reassesment every 5 periods: {} iterations, took {} seconds".format(counter_how11,round(VFI_time_how11,2)))
ax.plot(V[-1]-V_how12[-1],color='red',label="Howard's with reassesment every 10 periods: {} iterations, took {} seconds".format(counter_how12,round(VFI_time_how12,2)))
ax.plot(V[-1]-V_how13[-1],color='gray',label="Howard startting at 20: {} iterations, took {} seconds".format(counter_how13,round(VFI_time_how13,2)))
ax.plot(V[-1]-V_how14[-1],color='green',label="Howard startting at 50: {} iterations, took {} seconds".format(counter_how14,round(VFI_time_how14,2)))
ax.set_title("Distance between functions")
ax.set_xlabel("number of iterations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()