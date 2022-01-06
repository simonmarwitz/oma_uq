'''
Created on Mar 30, 2020

@author: womo1998
'''
import numpy as np
import matplotlib.pyplot as plot

def get_deflection_point(alpha,xi, F,l, E,I):
    
    beta = 1-alpha
    xi_ = 1-xi
    
    if xi <=alpha:
        return (1-beta**2-xi**2)*beta*xi*F*l**3/6/E/I
    else:
        return (1-alpha**2-xi_**2)*alpha*xi_*F*l**3/6/E/I


def get_deflection_curve(npoints, alpha, F, l, E, I):
    
    xi_vec = np.linspace(0,1,npoints,True)
    w_vec = np.empty_like(xi_vec)
    
    for i in range(npoints):
        w_vec[i] = get_deflection_point(alpha, xi_vec[i], F, l, E, I)
    
    return w_vec


def main():
    npoints = 25
    alpha=.25
    F=1000
    l=10000
    E=11000
    I=80e6
    
    nsamples = 10000
    
    
    mu_F = 1000
    var_F = 40
    rand_vals= [[],[],[]]
    
    for E in [8000, 11000]:
        E_vec = np.random.rand(nsamples)*1000+E
        F_vec = np.random.randn(nsamples)*var_F+mu_F
        a_vec = np.random.randn(nsamples)*0.2+5
        
        max_w_monte_carlo = np.empty_like(F_vec)
        
        for i in range(nsamples):
            
            w_curve = get_deflection_curve(npoints, a_vec[i]/l, F_vec[i], l, E_vec[i], I)
            max_w_monte_carlo[i] = max(w_curve)
        rand_vals[0].append(F_vec)
        rand_vals[1].append(a_vec)
        rand_vals[2].append(max_w_monte_carlo)
    print(np.histogram(F_vec, density=True))
        
    fig, axes = plot.subplots(3,3, sharex='col')
    
    rand_labels = ['Force', 'Position', 'Deflection']
    for i in range(3):
        axes[i,i].hist(rand_vals[i], bins=50, cumulative=True, histtype='step')
        for j in range(i):
            axes[i,j].plot(rand_vals[j][0],rand_vals[i][0], ls='none',marker=',')
            axes[i,j].plot(rand_vals[j][1],rand_vals[i][1], ls='none',marker=',')
        axes[-1,i].set_xlabel(rand_labels[i])
        axes[i,-1].yaxis.set_label_position("right")
        axes[i,-1].set_ylabel(rand_labels[i],rotation='vertical',)
    plot.show()
            
#     plot.figure()
#     ax1 = plot.subplot()
#     plot.figure()
#     ax2 = plot.subplot()
#     
#     for npoints in [3,5,25,100,1000]:
#     
#         l_vec = np.linspace(0,l,npoints)
#         
#     #     w_curve = get_deflection_curve(npoints, alpha, F, l, E, I)
#     #     plot.plot(l_vec,-w_curve)
#     #     ind = np.argmax(w_curve)
#     #     plot.plot(l_vec[ind],-w_curve[ind], ls='none',marker='x')
#     #     plot.show()
#         num_exp = 100
#         max_w = np.empty((num_exp,))
#         a_vec = np.linspace(0,l,num_exp)
#         x_max = np.empty((num_exp,))
#         for i in range(num_exp):
#             alpha = a_vec[i]/l
#             w_curve = get_deflection_curve(npoints, alpha, F, l, E, I)
#     #         plot.plot(l_vec,-w_curve)
#     #         plot.show()
#             ind = np.argmax(w_curve)
#             max_w[i]=w_curve[ind]
#             x_max[i] = l_vec[ind]
#         ax1.plot(a_vec, max_w)
# 
#         ax2.plot(a_vec, x_max)
#     plot.show()
        
    

if __name__ == '__main__':
    
    main()