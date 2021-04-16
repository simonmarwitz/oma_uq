import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize

def newmark_ampli_physical(k, c,m, deltat, timesteps=None, 
                  gamma_ =0.5, beta_=0.25, alpha_f_=0, alpha_m_ = 0):
    frequencies, damping,num_cycles, mode=None,None,None,None
    return newmark_ampli(frequencies, damping, k, c, m, deltat, timesteps, num_cycles, mode, gamma_, beta_, alpha_f_, alpha_m_)


def newmark_ampli(frequencies=None, damping=None, k=None, c=None,m=None, deltat=None, timesteps=None, num_cycles=10, mode=0,
                  gamma_ =0.5, beta_=0.25, alpha_f_=0, alpha_m_ = 0):
    
    
    mode=0
    if frequencies is not None and damping is not None:
        zeta_ = damping[mode]
        freq_ = frequencies[mode]
        omega_ = freq_*2*np.pi
        Omega_ = omega_*deltat
        A1=np.array([[1,                      0,                           -beta_*deltat**2],
                     [0,                      1,                           -gamma_*deltat  ],
                     [omega_**2*(1-alpha_f_), 2*zeta_*omega_*(1-alpha_f_), 1-alpha_m_      ]])
        A2=np.array([[1,                     1*deltat,                   (0.5-beta_)*deltat**2],
                     [0,                     1,                          (1-gamma_)*deltat    ],
                     [-omega_**2*(alpha_f_), -2*zeta_*omega_*(alpha_f_), -alpha_m_            ]])
        
    elif k is not None and c is not None and m is not None:
    
        A1=np.array([[1,              0,              -beta_*deltat**2],    #d_n+1
                     [0,              1,              -gamma_*deltat  ],    #v_n+1
                     [k*(1-alpha_f_), c*(1-alpha_f_), m*(1-alpha_m_)  ]])   #a_n+1
#         A1=np.fliplr(A1)
    
        A2=np.array([[1,             1*deltat,      (0.5-beta_)*deltat**2], #d_n
                     [0,             1,             (1-gamma_)*deltat    ], #v_n
                     [-k*(alpha_f_), -c*(alpha_f_), -m*alpha_m_          ]])#a_n
#         A2=np.fliplr(A2)
        
    A=np.linalg.solve(A1,A2)
    lamda = np.linalg.eigvals(A)
    lamda=lamda[np.logical_and(np.real(lamda)!=0, np.imag(lamda)!=0)]
    # find the conjugate eigenvalues
    if np.logical_and(np.real(lamda)<0, np.imag(lamda)==0).all(): 
        print("System has only negative, real eigenvalues!")
        frequencies_n=np.nan
        damping_n = np.nan
    else:
        loglamda=np.log(lamda)
        if (np.imag(loglamda)>0).any():
            this_loglamda = loglamda[loglamda.imag>0].max()
            rho = np.abs(lamda)
            phi = np.abs(np.angle(lamda))
            j = np.argmax((phi < np.pi)*rho)
            Omega_hat = np.sqrt(phi[j]**2+np.log(rho[j])**2)
            zeta_hat=-np.log(rho[j])/Omega_hat
    #                     print(Omega_hat, zeta_hat)
            Omega_hat_dd=np.imag(this_loglamda)                    # numerically damped sampling frequency for damped response frequency
#             Omega_hat_du=Omega_hat_dd/np.sqrt(1-damping[mode]**2)  # numerically damped sampling frequency for undamped response frequency
            Omega_hat_ud=np.abs(this_loglamda)                     # numerically undamped sampling frequency for damped response frequency 
                                                                   # the former may be for the UNdamped response frequency -> see below
#             Omega_hat_uu=Omega_hat_ud/np.sqrt(1-damping[mode]**2)  # numerically undamped sampling frequency for undamped response frequency
            
            # All but the first (dd) have a constant offset in frequency error, when a damped system is used
            
                
            zeta_hat=-np.real(this_loglamda)/Omega_hat_ud # contains both physical and numerical damping

#             Omega_hat_dd_=Omega_hat_ud*np.sqrt(1-zeta_hat**2)      # numerically damped sampling frequency for damped response frequency 
                                                                   # interestingly it matches Omega_hat_dd, I would have thought, since 
                                                                   #zeta_hat contains all damping, we would have to use Omega_hat_uu in the first term 
            
            frequencies_n = Omega_hat_dd/2/np.pi/deltat
            damping_n = zeta_hat
    
    

        
    time=np.zeros((timesteps,))
    resp=np.zeros((timesteps,3))
    R_init=1
    if frequencies is not None and damping is not None:
        a_init=-R_init*omega_**2
    elif k is not None and c is not None and m is not None:
        a_init=-R_init*k/m
    resp[0,:]=A.dot([R_init,0,a_init])
    time[0]=deltat
    for i in range(1, timesteps):
        resp[i]=A.dot(resp[i-1])
        time[i]=time[i-1]+deltat
    
    return time, np.fliplr(resp), frequencies_n, damping_n

def newmark_direct( K=None, C=None,M=None, deltat=None, timesteps=None,
                  gamma_ =0.5, beta_=0.25, alpha_f_=0, alpha_m_ = 0):
    
    omega_ = np.sqrt(K/M) # undamped omega
    zeta_ = C/2/np.sqrt(K*M)
    omega_*=np.sqrt(1-zeta_**2) # damped omega


    # If here the damped omega is provided, later Omega_hat_ud has to be used
    # If here the undamped omega is provided, later Omega_hat_dd has to be used
    # Providing damped omega is more logical, as this is the response frequency of the system, 
    # also this would be inline with providing the physical parameters for the amplification matrix
    A1=np.array([[1,                      0,                           -beta_*deltat**2],
                 [0,                      1,                           -gamma_*deltat  ],
                 [omega_**2*(1-alpha_f_), 2*zeta_*omega_*(1-alpha_f_), 1-alpha_m_      ]])
    A2=np.array([[1,                     1*deltat,                   (0.5-beta_)*deltat**2],
                 [0,                     1,                          (1-gamma_)*deltat    ],
                 [-omega_**2*(alpha_f_), -2*zeta_*omega_*(alpha_f_), -alpha_m_            ]])
    
#     A1=np.array([[1,              0,              -beta_*deltat**2],    #d_n+1
#                  [0,              1,              -gamma_*deltat  ],    #v_n+1
#                  [K*(1-alpha_f_), C*(1-alpha_f_), M*(1-alpha_m_)  ]])   #a_n+1
#     A1=np.fliplr(A1)
# 
#     A2=np.array([[1,             1*deltat,      (0.5-beta_)*deltat**2], #d_n
#                  [0,             1,             (1-gamma_)*deltat    ], #v_n
#                  [-K*(alpha_f_), -C*(alpha_f_), -M*alpha_m_          ]])#a_n
#     A2=np.fliplr(A2)
    
    a0=(1-alpha_m_)/(beta_*deltat**2)
    a1=(1-alpha_f_)*gamma_/(beta_*deltat)
    a2=a0*deltat
    a3=(1-alpha_m_)/(2*beta_)-1
    a4=(1-alpha_f_)*gamma_/beta_-1
    a5=(1-alpha_f_)*(gamma_/(2*beta_)-1)*deltat
     
     
    a0n=1/(beta_*deltat**2)
    a1n=gamma_/(beta_*deltat)
    a2n=a0n*deltat
    a3n=1/(2*beta_)-1
    a4n=gamma_/beta_-1
    a5n=(gamma_/(2*beta_)-1)*deltat
     
    M_star = a0*M+a1*C+(1-alpha_f_)*K
#     
#     A1=np.array([[1,         0,         -a0n                 ],      #a_i+1
#                  [0,         1,         -a1n                 ],      #v_i+1
#                  [0,         0,         M_star               ]])     #d_i+1
#     A2=np.array([[-a3n,      -a2n,      -a0n                 ],      #a_i
#                  [-a5n,      -a4n,      -a1n                 ],      #v_i
#                  [M*a3+C*a5, M*a2+C*a4, -K*alpha_f_+M*a0+C*a1]])      #d_i
    
    A=np.linalg.solve(A1,A2)
    lamda = np.linalg.eigvals(A)
    lamda=lamda[np.logical_and(np.real(lamda)!=0, np.imag(lamda)!=0)]
    # find the conjugate eigenvalues
    if np.logical_and(np.real(lamda)<0, np.imag(lamda)==0).all(): 
        print("System has only negative, real eigenvalues!")
        frequencies_n=np.nan
        damping_n = np.nan
    else:
        loglamda=np.log(lamda)
        if (np.imag(loglamda)>0).any():
            this_loglamda = loglamda[loglamda.imag>0].max()
            rho = np.abs(lamda)
            phi = np.abs(np.angle(lamda))
            j = np.argmax((phi < np.pi)*rho)
            Omega_hat = np.sqrt(phi[j]**2+np.log(rho[j])**2)
            zeta_hat=-np.log(rho[j])/Omega_hat
            Omega_hat_dd=np.imag(this_loglamda)                    # contains damping twice, as the damped sampling frequency was provided to the amplification matrix
            Omega_hat_ud=np.abs(this_loglamda)                     # undamped sampling frequency for damped response frequency, contains numerical damping 
            zeta_hat=-np.real(this_loglamda)/Omega_hat_ud          # contains both physical and numerical damping
            frequencies_n = Omega_hat_ud/2/np.pi/deltat
            damping_n = zeta_hat
    
    

        
    time=np.zeros((timesteps,))
    respd=np.zeros((timesteps,3))
    respa=np.zeros((timesteps,3))
    
    time[0]=deltat



    
    
#     a0n=a0
#     a1n=a1
#     a2n=a2
#     a3n=a3
#     a4n=a4
#     a5n=a5
    
    ampli=0
    direct=True
    
    d,v,a=[1,0,0]
    
    
    if direct:
        a=-K*d/M
        rhs=-alpha_f_*K*d+M*(a0*d+a2*v+a3*a)+C*(a1*d+a4*v+a5*a)
        dnext=rhs/M_star
        vnext=a1n*(dnext-d)-a4n*v-a5n*a
        anext = a0n*(dnext-d)-a2n*v-a3n*a
        respd[0]=[anext,vnext,dnext]
    if ampli:
        respa[0]=A.dot([a,v,d])
             
#     print(respd[0], respa[0])
    
    for i in range(1, timesteps):
        
        time[i]=time[i-1]+deltat
        if direct:
            a,v,d=respd[i-1]
            rhs=-alpha_f_*K*d+M*(a0*d+a2*v+a3*a)+C*(a1*d+a4*v+a5*a)
            dnext=rhs/M_star
            vnext=a1n*(dnext-d)-a4n*v-a5n*a
            anext = a0n*(dnext-d)-a2n*v-a3n*a
            respd[i]=[anext,vnext,dnext]
        if ampli:
            respa[i]=A.dot(respa[i-1])
        
#         print(respd[i], respa[i])
    
    if ampli and direct:
        
        plot.plot(respd-respd)
        plot.legend()
        plot.show()
    if ampli>direct:
        resp = respa
    elif direct:
        resp=respd
    elif ampli:
        resp=respa

    return time, resp, frequencies_n, damping_n


def free_decay(t, R, zeta, omega_d, phi=0):
#     R=1
#     phi=0
    #return -2*R*omega_d**2*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)
    return R*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)

def response_frequency(time_values, ydata, p0=[1,0.05,2*np.pi,0]):
#     res=pyansys.read_binary(path)
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
# 
#         
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#     
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
#     #print(nnum)
# 
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values
    
    popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = this_t, ydata=ydata, p0=p0)#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
    perr = np.sqrt(np.diag(pcov))
    #print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
    return popt, perr

def accuracy():
    frequencies=np.array([1])/np.pi
    damping = np.array([0.1])
    
    num_cycles=10
    f_min = frequencies[0]
    mode=0
    
    omega = frequencies[mode]*2*np.pi
    zeta=damping[mode]
    m=1
    k = omega**2*m/(1-zeta**2)# N/m
    d = 2*zeta*np.sqrt(k*m)
    df=0.05
    rho=(-df+1)/(df+1)    
    rho=0.5
    meths         =['NMK','NMK','HHT','WBZ','G-alpha']
    parameter_sets=['AAM','LAM',rho,rho,rho ]
    
    fig, axes = plot.subplots(2,2,sharey=True)  
    for j in range(5):
        meth= meths[j]
        parameter_set =parameter_sets[j]
        
        gamma_, beta_, alpha_f_, alpha_m_,_ = trans_params(meth,parameter_set)
        
        deltats=np.logspace(-3,-2,20)*3/frequencies[mode]
        frequencies_n = np.empty_like(deltats)
        dampings_n = np.empty_like(deltats)
        frequencies_id = np.empty_like(deltats)
        dampings_id = np.empty_like(deltats)
        
        for i in range(deltats.size):
            deltat = deltats[i]
            timesteps = int(np.ceil(num_cycles/f_min/deltat))
            if False:
                time, resp, frequencies_n[i], dampings_n[i] = newmark_ampli(
                                frequencies=frequencies, damping=damping, deltat=deltat, timesteps=timesteps, num_cycles=num_cycles, mode=mode, 
                                gamma_=gamma_, beta_=beta_, alpha_f_=alpha_f_, alpha_m_=alpha_m_)
            elif True:
                time, resp, frequencies_n[i], dampings_n[i] = newmark_direct(k, d, m, deltat, timesteps, gamma_, beta_, alpha_f_, alpha_m_)
            else:
                time, resp, frequencies_n[i], dampings_n[i] = newmark_ampli_physical(k, d, m, deltat, timesteps, gamma_, beta_, alpha_f_, alpha_m_)
            
            popt, _ = response_frequency(time, resp[:,0], p0=[1,damping[mode],frequencies[mode]*2*np.pi,0])
            frequencies_id[i]=popt[2]/2/np.pi
            dampings_id[i] = popt[1]
            
        periods_id=1/frequencies_id
        periods_n=1/frequencies_n
        periods=1/1
        h=deltats
        h=h[:20]
       
        
        color=axes[0,0].semilogy(periods_n,h/periods, label=f'{meth}_comp', ls='dotted',marker="+")[0].get_color()
        axes[0,0].semilogy(periods_id,h/periods, label=f'{meth}_id', color=color, ls='dashed',marker="x")
        axes[1,0].semilogy(dampings_n,h/periods, color=color, ls='dotted',marker="+")
        axes[1,0].semilogy(dampings_id,h/periods, color=color, ls='dashed',marker="x")
        axes[0,1].semilogy((periods_id-periods_n)/periods*100,h/periods,  color=color, ls='dashed',marker=".")
        axes[1,1].semilogy((dampings_id-dampings_n)*100,h/periods, color=color, ls='dotted',marker=".")
    axes[0,1].axvline(-0.1, color='lightgray')
    axes[0,1].axvline(0.1, color='lightgray')
    axes[1,1].axvline(-0.1, color='lightgray')
    axes[1,1].axvline(0.1, color='lightgray')
    
    
    mi,ma = axes[0,1].get_xlim()
    axes[0,1].set_xlim((min(mi,-ma),max(ma,-mi)))
    mi,ma = axes[1,1].get_xlim()
    axes[1,1].set_xlim((min(mi,-ma),max(ma,-mi)))
    plot.figlegend()
    
    plot.suptitle(f"$\\rho = {rho}$, $\\zeta = {0 if zeta is None else zeta}$")
    plot.show()
    

def trans_params(meth, parameter_set):
    
    delta=0 #gamma
    alpha=0 #beta
    alphaf=0 #alphaf
    alpham=0 #alpham
    
    if meth == 'NMK':
        tintopt='NMK'
        if isinstance(parameter_set, tuple):
            assert len(parameter_set) == 2
            delta, alpha = parameter_set
        elif isinstance(parameter_set, str):
            if parameter_set == 'AAM':# Trapezoidal Rule -> Constant Acceleration
#                 print('Using AAM parameters for NMK integration')
                delta = 1/2 
                alpha = 1/4
            elif parameter_set == 'LAM': # Original Newmark 1/6 -> Linear Acceleration
#                 print('Using LAM parameters for NMK integration')
                delta=1/2
                alpha=1/6
            elif parameter_set == 'CDM': # Explicit Central Difference Method
#                 print('Using CDM parameters for NMK integration. Warning: this will most likely fail!')
                delta=1/2
                alpha=0 
            else:
#                 print("Using default parameters (linear acceleration) for NMK integration.")
                delta=1/2
                alpha = 1/6
        elif isinstance(parameter_set, float):
#             print('Using custom parameters for NMK integration')
            rho_inf = parameter_set
            alphaf = 0
            alpham = 0
            delta = (3-rho_inf)/(2*rho_inf+2)
            alpha = 1/((rho_inf+1)**2)
        else:
#             print("Using default parameters (linear acceleration) for NMK integration.")
            delta=1/2
            alpha = 1/6
        
        
    elif meth == 'HHT':# HHT-α Hilber-Hugh Taylor
        tintopt='HHT'
        rho_inf = parameter_set
#         print(f'Using rho_inf {rho_inf} for HHT integration')
        alphaf = (1-rho_inf)/(rho_inf+1)
        alpham = 0
        delta = 1/2+alphaf
        alpha = (1+alphaf)**2/4
    elif meth == 'WBZ': # WBZ-α Wood-Bosak-Zienkiewicz 
        tintopt='HHT'
        rho_inf = parameter_set
#         print(f'Using rho_inf {rho_inf} for WBZ integration')
        alphaf = 0
        alpham = (rho_inf-1)/(rho_inf+1)
        delta = 1/2-alpham
        alpha = (1-alpham)**2/4
    elif meth == 'G-alpha':
        tintopt='HHT'
        rho_inf = parameter_set
#         print(f'Using rho_inf {rho_inf} for G-alpha integration')
        alpham = (2*rho_inf-1)/(rho_inf+1)
        alphaf = rho_inf/(rho_inf+1)
        delta = 1/2-alpham+alphaf
        alpha = 1/4*(1-alpham+alphaf)**2
    else:
        tintopt='HHT'
#         print('Using custom parameters for G-alpha integration')
        assert len(parameter_set) == 4
        delta, alpha, alphaf, alpham = parameter_set
    # -> gamma, beta, alphaf, alpham
    print(f'The following parameter have been computed for method {meth} with parameters {parameter_set}: gamma {delta}, beta {alpha}, alpha_f {alphaf}, alpha_m {alpham}')
    return delta, alpha, alphaf, alpham, tintopt

def main():
    frequencies=np.array([1/2/np.pi*2])
    damping = np.array([0.1])
    
    mode=0
    omega = frequencies[mode]*2*np.pi
    zeta=damping[mode]
    m=10
    k = omega**2*m/(1-zeta**2)# N/m
    d = 2*zeta*np.sqrt(k*m)
    
    deltat = 0.03
    timesteps=None
    num_cycles=10
    
    f_min = frequencies[0]
    if timesteps is None:
        timesteps = int(np.ceil(num_cycles/f_min/deltat))
    elif num_cycles is None:
        num_cycles = int(np.floor(timesteps*f_min*deltat))
    
    mode=0
    
    rho=0.714
    
    meths         =['NMK','NMK','HHT','WBZ','G-alpha']
    parameter_sets=['AAM','LAM',rho,rho,rho ]
    j=0
    gamma_, beta_, alpha_f_, alpha_m_,_ = trans_params(meths[j], parameter_sets[j])
    
#     time, resp, frequencies_n, damping_n = newmark_ampli(
#                     frequencies=frequencies, damping=damping, deltat=deltat, timesteps=timesteps, num_cycles=num_cycles, mode=mode, 
#                     gamma_=gamma_, beta_=beta_, alpha_f_=alpha_f_, alpha_m_=alpha_m_)
    time, resp, frequencies_n, damping_n = newmark_direct(k, d, m, deltat, timesteps, gamma_, beta_, alpha_f_, alpha_m_)
#     time, resp, frequencies_n, damping_n = newmark_ampli_physical(k, d, m, deltat, timesteps, gamma_, beta_, alpha_f_, alpha_m_)
    phi_n = np.arctan((-damping_n*frequencies_n*2*np.pi/np.sqrt(1-damping_n**2)*1)/(1*frequencies_n*2*np.pi))
    phi_ex = np.arctan((-damping[mode]*frequencies[mode]*2*np.pi/np.sqrt(1-damping[mode]**2)*1)/(1*frequencies[mode]*2*np.pi))
    
    try:
        popt, _ = response_frequency(time, resp[:,2], p0=[1,damping[mode],frequencies[mode]*2*np.pi,0])
    except Exception as e:
        print(e)
        popt=[1,1,0,0]
    frequencies_id=popt[2]/2/np.pi
    damping_id = popt[1]
    a_init, v_init, d_init= resp[0]
    phi_id = frequencies_id*deltat+np.arctan(-(v_init-damping_id*frequencies_id*2*np.pi/np.sqrt(1-damping_id**2)*d_init)/(d_init*frequencies_id*2*np.pi))
    
    time_fine = np.arange(0,deltat*timesteps, 0.0001)
#     
# #     deltat_= popt[3]/(frequencies[mode]*2*np.pi)
    print(*popt)
    for i,label in enumerate(["a","v","d"]):
        plot.plot(time,resp[:,i], marker='x',label=label)
    plot.plot(time_fine, free_decay(time_fine,1, damping[mode], frequencies[mode]*2*np.pi,phi_ex), label='ex')
    plot.plot(time_fine, free_decay(time_fine,1, damping_n, frequencies_n*2*np.pi,phi_n), label='est')
    plot.plot(time_fine, free_decay(time_fine,*popt), label='id')
    plot.legend()
    plot.show()

    
if __name__ == '__main__':
#     main()
    accuracy()