import numpy as np
import matplotlib.pyplot as plt



tau = 0.0056
#tau = 0.000346
#tau = 0.0025


tau2 = tau**2

N = 500

C=1;

X=2/tau2*(1-np.cos((2*np.linspace(0,C*N-1,C*N)+1)/2/(C*N)*np.pi)) #Chebyshev knots

w_min = 12
w_max = 12.6

lammin = w_min**2
lammax = w_max**2


#evaluate basis at chebyshev knots
Q = np.zeros((len(X),N))
Q[:,0] = np.ones(len(X))

#Q[:,1] = np.ones(len(X))
Q[:,1] = (2-tau2*X)*np.ones(len(X))/2
for k in range(2,N):
    Q[:,k] = (2-tau2*X)*Q[:,k-1] - Q[:,k-2]

#goalfunc = lambda x: np.exp(-1/50*(x-12.3**2)**2)
goalfunc = lambda x: (x<=lammax)*(x>=lammin)

val=goalfunc(X)
alpha=np.linalg.solve(Q,val)
# alpha=np.linalg.lstsq(Q,val)[0]

#Y=np.linspace(0,4/tau2,50000)
Y=np.linspace(0,400,50000)


#evaluate basis at knots for plotting
W = np.zeros((len(Y),N))
W[:,0] = np.ones(len(Y))
W[:,1] = (2-tau2*Y)*np.ones(len(Y))/2
for k in range(2,N):
    W[:,k] = (2-tau2*Y)*W[:,k-1] - W[:,k-2]
    
beta = W@alpha

ts = np.arange(N)*tau

#reference alphas computed as in preprint
alpharef =  tau*4/np.pi*np.cos((w_min+w_max)/2*ts)*(w_max-w_min)/2*np.sinc((w_max-w_min)/2*ts/np.pi)
betaref = W@alpharef


plt.plot(np.sqrt(Y),beta,label = 'interpolation')
plt.plot(np.sqrt(Y),betaref,label ='inverse fourier method')
plt.plot(np.sqrt(Y),goalfunc(Y),label = 'goal function')
#plt.plot(np.sqrt(X[:100]),np.zeros(len(X[:100])),'x',label = 'chebyshev knots')
plt.plot(np.sqrt(X[:100]),goalfunc(X[:100]),'x',label = 'chebyshev knots')
#plt.plot(np.sqrt(X),goalfunc(X),'x',label = 'chebyshev knots')
plt.legend()

plt.show()
