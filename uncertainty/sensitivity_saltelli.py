import numpy as np
np.set_printoptions(precision=6,suppress=True)

import scipy.stats
import matplotlib.pyplot as plot

def Y(Z,Omega):
    return np.sum(Z*Omega)

def example_chap_1():
    r=4
    N=1000
    
#     mu_z=np.zeros((r,))
#     sigma_z=np.ones((r,))*2    
#     
#     mu_Omega=np.array([4,3,2,1])
#     sigma_Omega=np.arange(1,r+1)
#     
#     sigma_Y = np.sqrt(np.sum(mu_Omega**2*sigma_z**2))
#     S_Z=np.zeros((r,))
#     for i in range(r):
#         S_Z[i] = (sigma_z[i]/sigma_Y*mu_Omega[i])
#     print('Sigma-normalized derivative',S_Z**2)
#     
#     M=np.zeros((N,r))
#     
#     for i in range(r):
#         M[:,i]=np.random.normal(mu_z[i],sigma_z[i],N)
#     Y=np.sum(M*mu_Omega,axis=1)
#     
#     print(f'sigma_Y {sigma_Y}, hat(sigma_Y) {np.std(Y)}')
#     
#     beta_Z = np.zeros((r,))
#     for i in range(r):
#         bzi,b0i,_,_,_ = scipy.stats.linregress(M[:,i],Y)
#         beta_Z[i] = bzi*sigma_z[i]/sigma_Y
#     
#     print('Regression sensitivies', beta_Z**2)
#     
#     fig,axes = plot.subplots(nrows=2, ncols=2, sharey=True)
#     axes = axes.flatten()
#     for i in range(r):
#         axes[i].plot(M[:,i],Y, ls='none', marker='.',markersize=1)
    

    mu_z=np.zeros((r,))
    sigma_z=np.arange(1,r+1)
    
    mu_Omega=np.arange(1,r+1)*0.5
    sigma_Omega=np.arange(1,r+1)
    
    sigma_Y2 = np.sqrt(np.sum(sigma_z**2*(sigma_Omega**2+mu_Omega**2)))#-np.sum(mu_z**2*mu_Omega**2))#+np.product(np.hstack((mu_Omega,mu_z))))
    
    S_Z2=np.zeros((2*r,))
    for i in range(r):
        S_Z2[i] = (sigma_z[i]/sigma_Y2*mu_Omega[i])
        S_Z2[i+r] = (sigma_Omega[i]/sigma_Y2*mu_z[i])
    print('Sigma-normalized derivatives ',S_Z2**2)
    print('"Degree of additivity" (p. 23) ',np.sum(S_Z2**2))
    
    M=np.zeros((N,r))
    for i in range(r):
        M[:,i]=np.random.normal(mu_z[i],sigma_z[i],N)
        
    M_Omega =np.zeros((N,r))
    for i in range(r):
        M_Omega[:,i]=np.random.normal(mu_Omega[i],sigma_Omega[i],N)
        
    Y2=np.sum(M*M_Omega,axis=1)
    #print(f'sigma_Y2 {sigma_Y2}, hat(sigma_Y2) {np.std(Y2)}')
    
    all_M = np.hstack((M, M_Omega))
    
    fig,axes = plot.subplots(nrows=3, ncols=3, sharey=True)
    axes = axes.flatten()
    for i in range(2*r):
        axes[i].plot(all_M[:,i],Y2, ls='none', marker='.',markersize=1)
    
    all_sigma = np.hstack((sigma_z, sigma_Omega))
    beta_Z2 = np.zeros((2*r,))
    for i in range(2*r):
        bzi,b0i,_,_,_ = scipy.stats.linregress(all_M[:,i],Y2)
        beta_Z2[i] = bzi*all_sigma[i]/sigma_Y2
    
    print('Regression sensitivies', beta_Z2**2)

#     plot.show()
    for i in range(r):
        print(np.sqrt(sigma_z[i]**2*sigma_Omega[i]**2+sigma_z[i]**2*mu_Omega[i]**2+mu_Omega[i]**2*sigma_Omega[i]**2)/sigma_Y2-S_Z2[i]-S_Z2[i+r])
                            
def main():
    #import chaospy
    import scipy.stats.qmc
    
    
    
    num_distributions=10
    
    engine = scipy.stats.qmc.Halton(num_distributions, scramble=False)
    engine.fast_forward(1) # skip first point = [0,...,0]
    samples = engine.random(1000).T
    
    print(samples.shape)
    
    for i in range(num_distributions):
        samples[i,:] = scipy.stats.arcsine(0,1).ppf(samples[i,:])
        
        
        pass
        
    #distribution = chaospy.J(*[chaospy.Normal(0, 1) for i in range(num_distributions)])
    #samples = distribution.sample(100, rule='halton')
    
    #for i in range(num_distributions):
    #    np.random.shuffle(samples[i,:]) # avoid structures (correlations) in scatter plots, while keeping low discrepancy, not sure if that is actually valid (saltelli p 87)
    #print(samples.shape, type(samples))
    products = np.zeros((num_distributions,num_distributions))
    fig, axes = plot.subplots(nrows=num_distributions, ncols=num_distributions)
    for i in range(num_distributions):
        for j in range(num_distributions):
            if i == j:
                products[i,j] = np.mean(samples[i,:])
                axes[i,j].hist(samples[i,:],bins=20)
            else:
                products[i,j] = np.mean(samples[i,:]*samples[j,:])
                axes[i,j].plot(samples[i,:],samples[j,:], ls='none', marker=',')
        
                #axes[i,j].set_ylim((-3,3))
                #axes[i,j].set_xlim((-3,3))
            axes[i,j].xaxis.set_visible(False)
            axes[i,j].yaxis.set_visible(False)
                
    plot.subplots_adjust(wspace=0, hspace=0, top=1,bottom=0, left=0, right=1)
    
    #plot.figure()
    plot.matshow(products,vmin=0, vmax=1)
    plot.colorbar()

    plot.show()

if __name__ == '__main__':
    main()