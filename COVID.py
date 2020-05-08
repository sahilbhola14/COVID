#################################################
#Developed by: Sahil Bhola			#
#Contributor: Kishore Premkumar			#
#Mail: sbhola@umich.edu				#
#Please direct all your queries through mail	#
#################################################
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy.io as sio
import scipy.linalg as LA
from scipy.spatial import distance
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Dict(c): Cases: US(0),Italy(1),UK(2),Spain(3),Germany(4),Turkey(5),Austria(6),Australia(7),Norway(8),NZ(9),France(10),SK(11)
# Dict(m): Model: SIR(0),SIRT(1)
# cases_tot: [22-01-2020 to 5-05-2020]
c = 11; #select country
m = 1; #Model ID
lambda1 = 0	#Regularization on Infected cases
lambda2 = 1e-3 #Regularization on Recovered cases
iLayers = np.array([1,10,1])	#NN Layer parameters;[x,y,z],x:input,y:Hidden layer,z:output
tt = 0.5	#Percentage of data trained

def data(c):
	#Input: Country ID
        #Output: Array of Infected cases(>=500),Recovered cases, population size
        F1 = loadmat('data.mat')
        cases_I = F1['Infected']
        cases_R = F1['Recovered']
        population = F1['pop']
        return cases_I[c,cases_I[c,:]>=500],cases_R[c,cases_I[c,:]>=500],population[0,c] # When infected cases > 500

def Solve_SIR(U,t,D):
	N = population*1e6
	S = U[0]; I = U[1]; R = U[2]
	beta = D[-2]
	gamma = D[-1]
	dSdt = -beta*S*I/N
	dIdt = (beta*S*I/N) -gamma*I
	dRdt = gamma*I

	return [dSdt,dIdt,dRdt]


def ODE_SIR(D):
	x = odeint(Solve_SIR,Uo,t,args=(D,))

	return x

def SIR(D):
	global count,f1
	x = ODE_SIR(D)
	MSE = LA.norm(np.log(x[t,1])-np.log(cases_I[t]))**2 + LA.norm(np.log(x[t,2])-np.log(cases_R[t]))**2 + lambda1*LA.norm(np.log(x[t,1]))**2+ lambda2*LA.norm(np.log(x[t,2]))**2
	count+=1
	f1.write('{0:.16f} {1:.16f}\n'.format(count,MSE))
	print(MSE)

	return (MSE)

def ReLU(val):
	val = val.reshape(val.shape[0])
	act_val = np.copy(val)
	act_val[val>=0] = val[val>=0]
	act_val[val<0] = 0

	return act_val.reshape(act_val.shape[0],1)


def NN(D,t):
	P = np.copy(D[:-2])
	z = 0
	fx = np.copy(np.array(t).reshape(1,1))
	for i in range(1,iLayers.shape[0]):
		Rw = range(z+0,z+iLayers[i-1]*iLayers[i])
		z = z+iLayers[i-1]*iLayers[i]
		Rb = range(z+0,z+iLayers[i])
		z = z + iLayers[i]
		W = P[Rw].reshape(iLayers[i],iLayers[i-1])
		b = P[Rb].reshape(iLayers[i],1)
		act_fx = ReLU(np.dot(W,fx) + b)
		fx = np.copy(act_fx)

	return np.asscalar(fx)


def Solve_SIRT(U,t,D):
	N = population*1e6
	S = U[0]; I = U[1]; R = U[2]; T = U[3]
	beta = D[-2]
	gamma = D[-1]
	dSdt = -beta*S*I/N
	dIdt = (beta*S*I/N) -(gamma+NN(D,t))*I
	dRdt = gamma*I
	dTdt = NN(D,t)*I

	return [dSdt,dIdt,dRdt,dTdt]

def ODE_SIRT(D):
	x = odeint(Solve_SIRT,Uo,t,args=(D,))

	return x


def SIRT(D):
	global count,f1
	x = ODE_SIRT(D)
	MSE = LA.norm(np.log(x[:,1])-np.log(cases_I[t]))**2 + LA.norm(np.log(x[:,2])-np.log(cases_R[t]))**2 + lambda1*LA.norm(np.log(x[:,1]))**2+ lambda2*LA.norm(np.log(x[:,2]))**2
	count+=1
	f1.write('{0:.16f} {1:.16f}\n'.format(count,MSE))
	print(MSE)

	return MSE

def const(D):
	return [D[-2],D[-1]]

def optimize(D):
	bnds = np.tile(np.array([0,None]),63).reshape(63,2)
	ieq1 = {'type':'ineq','fun':lambda x:const(x)[0]}
	ieq2 = {'type':'ineq','fun':lambda x:const(x)[1]}
	ieq3 = {'type':'ineq','fun':lambda x:1-const(x)[0]}
	ieq4 = {'type':'ineq','fun':lambda x:1-const(x)[1]}
	if m ==0:
		res = minimize(SIR, D, method='trust-constr',jac=None,bounds=None,constraints=[ieq1,ieq2],options={'disp': True})
	elif m ==1:
		#res = minimize(SIRT, D, method='trust-constr',jac=None,bounds=None,constraints=[ieq1,ieq2,ieq3,ieq4],options={'disp': True})
		res = minimize(SIRT, D, method='Nelder-Mead',jac=None,bounds=None,options={'maxiter':None,'disp': True})
	return res

def main():
	global population,cases_I,cases_R,t,Uo,count,tTrain,f1
	count = 0
	param = 0
	for i in range(1,iLayers.shape[0]):
		param = param + iLayers[i]*iLayers[i-1]
		param = param + iLayers[i]
	D = np.zeros(param + 2)	#Initial design variables
	cases_I,cases_R,population = data(c)
	nTrain = int(tt*cases_I.shape[0])
	t = np.linspace(0,cases_I.shape[0]-1,nTrain).astype(int)
	t_predict = np.linspace(cases_I.shape[0],cases_I.shape[0]+31,30)
	## IC and model selection
	S = population*1e6; I = cases_I[0]; R = 10; T = 10;
	## Select model
	models = np.array([[S,I,R],[S,I,R,T]])
	Uo = models[m]
	## Optimizer
	with open('Residual_C_'+str(c)+'_M_'+str(m)+'_tt_'+str(tt)+'.dat','w+') as f1:
		f1.write('Vaiables = Iteration,MSE\n')
		f1.write('ZONE T = GRID I = 1000 J = 1\n')
		res = optimize(D)

	## Plotting
	D = res.x
	sio.savemat('Dvariables_C_'+str(c)+'_M_'+str(m)+'_tt_'+str(tt)+'.mat',{'D':D})
	if m==0:
		t = np.linspace(0,100)
		x = ODE_SIR(D)
	elif m==1:
		t = np.linspace(0,100)
		x = ODE_SIRT(D)

	S = x[:,0]; I = x[:,1]; R = x[:,2]; T = x[:,3]
	plt.figure(1)
	t_plot = np.linspace(0,cases_I.shape[0]-1,cases_I.shape[0]).astype(int)
	plt.plot(t,np.log(I), label = 'Cases Infected (Modelled)')
	plt.plot(t,np.log(R), label = 'Cases Recovered (Modelled)')
	plt.plot(t,np.log(T), label = 'Cases Quarantined (Modelled)')
	plt.scatter(t_plot,np.log(cases_I), label = 'Cases infected (Actual)')
	plt.scatter(t_plot,np.log(cases_R), label = 'Cases recovered (Actual)')
	plt.legend(loc='lower right')
	plt.title('SIRT Model (South Korea)')
	plt.yscale('log')
	plt.xlabel('Days since outbreak')
	plt.ylabel('Cases')
	#plt.show()
	plt.savefig('Plot_C_'+str(c)+'_M_'+str(m)+'_tt_'+str(tt)+'.png')

if __name__=='__main__':
	main()
