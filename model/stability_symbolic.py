# coding: utf-8

#get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
import matplotlib
import sympy
sympy.init_printing()
import sympy.matrices
#import sympy.plotting
import numpy as np
import sympy.plotting.experimental_lambdify

an1,an,dn1,vn1,dn,vn,=sympy.symbols('a_{n+1} a_n d_{n+1} v_{n+1} d_n v_n', real=True, positive=True)
h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
alpha_f, alpha_m, rho_inf=sympy.symbols('$\\alpha_f$ $\\alpha_m$ $\\rho_{\infty}$', real=True)
m,c,k = sympy.symbols('m c k', positive=True, real=True)

print_context_dict ={'text.usetex':True,
                     'text.latex.preamble':"\\usepackage{siunitx}\n \\usepackage{xfrac}",
                     'font.size':10,
                     'legend.fontsize':10,
                     'xtick.labelsize':10,
                     'ytick.labelsize':10,
                     'axes.labelsize':10,
                     'font.family':'serif',
                     'legend.labelspacing':0.1,
                     'axes.linewidth':0.5,
                     'xtick.major.width':0.2,
                     'ytick.major.width':0.2,
                     'xtick.major.width':0.5,
                     'ytick.major.width':0.5,
                     'figure.figsize':(5.906,5.906/1.618),#print #150 mm \columnwidth
                     #'figure.figsize':(5.906/2,5.906/2/1.618),#print #150 mm \columnwidth
                     #'figure.figsize':(5.53/2,2.96),#beamer
                     #'figure.figsize':(5.53/2*2,2.96*2),#beamer
                     'figure.dpi':300}
    #figsize=(5.53,2.96)#beamer 16:9
    #figsize=(3.69,2.96)#beamer 16:9
    #plot.rc('axes.formatter',use_locale=True) #german months
# must be manually set due to some matplotlib bugs
if print_context_dict['text.usetex']:
    #plt.rc('text.latex',unicode=True)
    plt.rc('text',usetex=True)



# Newmark Stability Analysis
if False:
    #As1=sympy.matrices.Matrix([[1+beta*h**2*omega**2,h**2*beta*2*omega*zeta],[gamma*h*omega**2,1 + gamma*h*2*omega*zeta]])
    #As2=sympy.matrices.Matrix([[1-h**2/2*(1-2*beta)*omega**2,h-h**2*(0.5-beta)*2*omega*zeta],[-h*(1-gamma)*omega**2,1-2*h*(1-gamma)*zeta*omega]])
    #As=As1.solve(As2)

    #eigvals_s=list(As.subs(omega, Omega/h).eigenvals().keys())


    A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
    #with this one, Eq. 3.17 gives an oscillating response
    A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
    A=A1.solve(A2)

    #dn1=A.dot(sympy.Matrix([dn,vn,an]))[0]

    eigvals=list(A.subs(omega, Omega/h).eigenvals().keys())

#     p11=sympy.plotting.plot(abs(eigvals[1].subs(zeta,0).subs(beta,0).subs(gamma,0.5)),(Omega,0,10),ylim=(0,10),ylabel='$\\rho$', show=False)
#     p21=sympy.plotting.plot(abs(eigvals[2].subs(zeta,0).subs(beta,0).subs(gamma,0.5)),(Omega,0,10),ylim=(0,10),ylabel='$\\rho$', show=False)
#     p11.append(p21[0])
#     p11.show()
# 
#     p14=sympy.plotting.plot(abs(eigvals[1].subs(zeta,0).subs(beta,0.25).subs(gamma,0.5)),(Omega,0,10),ylim=(0,10),ylabel='$\\rho$', show=False)
#     p24=sympy.plotting.plot(abs(eigvals[2].subs(zeta,0).subs(beta,0.25).subs(gamma,0.5)),(Omega,0,10),ylim=(0,10),ylabel='$\\rho$', show=False)
#     p14.append(p24[0])
#     p14.show()
# 
#     sympy.plotting.plot3d(abs(eigvals[1].subs(zeta,0).subs(Omega,1)),(beta,0,2,),(gamma,0,3), zlabel='$\rho$', xlabel='$\\beta$', ylabel='$\\gamma$')._backend.ax[0].collections[0].set_cmap("tab20")
#     sympy.plotting.plot3d(abs(eigvals[2].subs(zeta,0).subs(Omega,1)),(beta,0,2,),(gamma,0,3), zlabel='$\rho$', xlabel='$\\beta$', ylabel='$\\gamma$')._backend.ax[0].collections[0].set_cmap("tab20")

    root=eigvals[0].subs(zeta,0)
    beta_bif=sympy.limit(sympy.solve(root.args[1].args[3],beta)[0],Omega,sympy.oo) # unconditional stability complex roots
    omega_bif=sympy.solve(root.args[1].args[3],Omega)[1] # bifurcation limit for conditional stability
    beta_lim = sympy.solve((1/beta*(gamma+sympy.sqrt(-beta+gamma**2))-2).subs(gamma, 1/2*(gamma+1/2)),beta)[0] # unconditional stability

    rho1=sympy.Abs((1-1/beta*(gamma+sympy.sqrt(-beta+gamma**2))).subs(gamma, 1/2*(gamma+1/2)))
    rho_lim = sympy.Abs(sympy.sqrt(1-(gamma-1/2)*eta**2).subs(eta**2, 1/beta))
    sympy.plotting.plot_implicit(rho1<0.1,(gamma,0,1),(beta,0,1))


# G-alpha stability
if False:
    h,beta,gamma,zeta,omega,Omega,eta, alpha_f, alpha_m=sympy.symbols('h \\beta \gamma \zeta \omega \\varOmega \eta \\alpha_f \\alpha_m')
    A1=sympy.matrices.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
    A2=sympy.matrices.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
    A=A1.solve(A2)

    D=1/A[0,0].args[0].args[0]
    AdD=sympy.simplify(A*D)


    Aoo = sympy.matrices.Matrix([[0,0,0],[0,0,0],[0,0,0]])

    for i in range(3):
        for j in range(3):
            Aoo[i,j]=sympy.limit(AdD[i,j]/D,Omega,sympy.oo)


# predictor corrector
if False:
    A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[0,0, m]])
    A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
    A=A1.solve(A2)

#Accuracy

# Newmark, no physical damping, manual
if 0:
    a = 1-1/2*(gamma+1/2)*eta**2
    b= eta*sympy.sqrt(-eta**2*1/4*(gamma+1/2)**2+1)
    lamda1 = a+b*sympy.I

    rho_d = sympy.sqrt(a**2+b**2)
    phi_d = abs(sympy.re(2*sympy.atan(b/(rho_d+a))))
    omega_bif=(1/4*(gamma+1/2)**2-beta)*(-1/2)

# Newmark, physical damping, manual
elif 0:
    A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
    A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
    A=A1.solve(A2)
    eigvals=list(A.subs(omega, Omega/h).eigenvals().keys())
    root=eigvals[0]#.subs(zeta,0)
    a=root.args[0]
    b=sympy.sqrt(root.args[1].args[3].args[0]*-1)*root.args[1].args[0]*root.args[1].args[1]*root.args[1].args[2]

    lamda1 = a+b*sympy.I

    rho_d = sympy.sqrt(a**2+b**2)
    phi_d = sympy.Abs(sympy.re(2*sympy.atan(b/(rho_d+a))))
    omega_bif = (-1/2*zeta*(-gamma+1/2)+sympy.sqrt(zeta**2*(beta-1/2*gamma)-beta+1/4*(gamma+1/2)**2))/(1/4*(gamma+1/2)**2-beta)

# Newmark, physical damping, automatic
elif True:
    A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
    A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
    A=A1.solve(A2)
    eigvals=list(A.subs(omega, Omega/h).eigenvals().keys())
    root=eigvals[0]#.subs(zeta,0) # taking the other eigenvalue in which the imaginary part is subtracted

    rho_d = sympy.Abs(root)
    phi_d = sympy.Abs(sympy.arg(root))
    omega_bif=sympy.solve(root.args[1].args[3],Omega)[1] # bifurcation limit for conditional stability

eta_ = Omega/sympy.sqrt(Omega**2*beta+1)

# omega_d = sympy.sqrt(phi_d**2+sympy.ln(rho_d)**2)
# zeta_d = -sympy.ln(rho_d)/omega_d

omega_d = -sympy.im(sympy.ln(root))
zeta_d = -sympy.re(sympy.ln(root))/sympy.Abs(sympy.ln(root))


if False:
    with matplotlib.rc_context(rc=print_context_dict):
        fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        x = np.linspace(0,10)[1:]
        for method, gamma_, beta_, linestyle in [('Average Acceleration',0.5,0.25,'solid'),
                                                 ('Linear Acceleration',0.5,1/6, 'dashed'),
                                                 ('Central Difference',0.5,0,'dotted'),
                                                 ('$\\gamma=0.6, \\beta=0.3025$',0.6,0.3025,'dashdot'),
                                                 ('$\\gamma=0.9142, \\beta=0.5$',np.sqrt(2)-1/2,1/2, (0, (3, 1, 1, 1, 1, 1))),]:

            for color, zeta_ in [('black',0), ('lightgray',0.1)]:
                #print(method, round(1/4*(gamma_+1/2)**2, 6), zeta_,omega_bif.subs(zeta,zeta_).subs(gamma,gamma_).subs(beta,beta_))
                if beta_ < round(1/4*(gamma_+1/2)**2,6): this_omega_bif = float(omega_bif.subs(zeta,zeta_).subs(gamma,gamma_).subs(beta,beta_))
                else: this_omega_bif = 100
                x = np.logspace(-2, np.log10(this_omega_bif))
                exp= sympy.plotting.experimental_lambdify.vectorized_lambdify([Omega],(rho_d.subs(eta, eta_).subs(gamma,gamma_).subs(beta,beta_).subs(zeta,zeta_)))
                if zeta_ == 0: label = method
                else: label = None
                
                axes[0].semilogx(x/np.sqrt(1-zeta_**2), exp(x),
                    label= label,
                    color = color, linestyle = linestyle)
                if this_omega_bif != 100:
                    shortm = {"Linear Acceleration":"LAM",
                              "Central Difference":"CDM",
                              }[method]
                    axes[0].plot(this_omega_bif, exp(this_omega_bif), marker='x', color=color)
                    an = axes[0].annotate("$\\Omega_{{\\mathrm{{bif}}}}^{{\mathrm{{{}}}}}$".format(shortm), (this_omega_bif, exp(this_omega_bif)), color=color)
                    an.draggable()
        axes[0].set_ylabel('$\\rho$')
        axes[0].set_xlabel('$\\Omega$')
        axes[0].set_ylim((0,1.05))
        axes[0].legend()

        #plt.show()

        zeta, beta, gamma, alpha_f, alpha_m = sympy.symbols('\\zeta \\beta \\gamma \\alpha_f \\alpha_m')

        A1=sympy.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
        A2=sympy.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
        A=A1.solve(A2)

        rho_inf = 0.55

        for method, linestyle in [('HHT-$\\alpha$','solid'),
                                 ('WBZ-$\\alpha$', 'dashed'),
                                 ('G-$\\alpha$','dotted'),
                                 ('Newmark-$\\beta$','dashdot'),]:

            alpha_f_, alpha_m_, gamma_, beta_ = None,None,None,None
            if method == 'Newmark-$\\beta$':
                alpha_f_ = 0
                alpha_m_ = 0
                gamma_ = (3-rho_inf)/(2*rho_inf+2)
                beta_ = 1/((rho_inf+1)**2)
            elif method == 'HHT-$\\alpha$':
                alpha_f_ = (1-rho_inf)/(rho_inf+1)
                alpha_m_ = 0
                gamma_ = 1/2+alpha_f_
                beta_ = (1+alpha_f_)**2/4
            elif method == 'WBZ-$\\alpha$':
                alpha_f_ = 0
                alpha_m_ = (rho_inf-1)/(rho_inf+1)
                gamma_ = 1/2-alpha_m_
                beta_ = (1-alpha_m_)**2/4
            elif method == 'G-$\\alpha$':
                alpha_m_ = (2*rho_inf-1)/(rho_inf+1)
                alpha_f_ = rho_inf/(rho_inf+1)
                gamma_ = 1/2-alpha_m_+alpha_f_
                beta_ = 1/4*(1-alpha_m_+alpha_f_)**2
            for color, zeta_ in [('black',0), ('lightgray',0.1)]:

                x = np.logspace(-2, 2)
                rho_d = np.empty_like(x)

                A_sub = A.subs(beta, beta_).subs(gamma, gamma_).subs(alpha_f, alpha_f_).subs(alpha_m,alpha_m_).subs(zeta, zeta_)
                #print(A_sub.free_symbols)
                for i in range(x.shape[0]):
                    lamdas = np.linalg.eigvals(np.array(A_sub.subs(Omega, x[i])).astype(np.float64))

                    rho = np.abs(lamdas)
                    phi = np.abs(np.angle(lamdas))
                    j = np.argmax((phi < np.pi)*rho)

                    rho_d[i] = rho[1]
                if zeta_ == 0: label = method
                else: label = None
                axes[1].semilogx(x/np.sqrt(1-zeta_**2), rho_d,
                    label=label,
                    color = color, linestyle = linestyle)

        axes[1].set_xlabel('$\\Omega$')

        #axes[1].set_ylabel('$\\rho$')
        axes[1].legend()
        axes[1].set_xlim((0.0105,100))
        plt.subplots_adjust(left=0.075, bottom=0.120, right=0.974, top=0.959, wspace=0.04)

        plt.show()


if False:
    with matplotlib.rc_context(rc=print_context_dict):

        fig,axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)
        gamma_=0.5
#         x = np.linspace(0,10)[1:]
        for method, gamma_, beta_, linestyle in [('Average Acceleration',0.5,0.25,'solid'),
                                                 ('Linear Acceleration',0.5,1/6, 'dashed'),
                                                 ('Central Difference',0.5,0,'dotted'),
                                                 ('$\\gamma=0.6, \\beta=0.3025$',0.6,0.3025,'dashdot'),
                                                 ('$\\gamma=0.9142, \\beta=0.5$',np.sqrt(2)-1/2,1/2, (0, (3, 1, 1, 1, 1, 1))),]:

            for color, zeta_ in [('black',0), ('lightgray',0.1)]:
                if beta_ < round(1/4*(gamma_+1/2)**2,6): this_omega_bif = float(omega_bif.subs(zeta,zeta_).subs(gamma,gamma_).subs(beta,beta_))
                else: this_omega_bif = 10
                x = np.logspace(-2, np.log10(this_omega_bif),50)
                exp= sympy.plotting.experimental_lambdify.vectorized_lambdify([Omega],(omega_d.subs(eta, eta_).subs(gamma,gamma_).subs(beta,beta_).subs(zeta,zeta_)))

                if zeta_ == 0: label = method
                else: label = None

                #x/=np.sqrt(1-zeta_**2)# comparing undamped periods
                axes[0].semilogy((x/exp(x)-1)*100, x/np.sqrt(1-zeta_**2)/2/np.pi,
                    label=label,
                    color = color, linestyle = linestyle)
        axes[0].set_xlabel('$\\sfrac{(\\hat{T} - T)}{T} \; [\\si{\\percent}]$')
        axes[0].set_ylabel('$\\sfrac{h}{T}$', labelpad=-15)
        axes[0].set_xlim((-35 ,170))
        #axes[0].legend()


#         x = np.linspace(0,10)[1:]
        for method, gamma_, beta_, linestyle in [('Average Acceleration',0.5,0.25,'solid'),
                                                 ('Linear Acceleration',0.5,1/6, 'dashed'),
                                                 ('Central Difference',0.5,0,'dotted'),
                                                 ('$\\gamma=0.6, \\beta=0.3025$',0.6,0.3025,'dashdot'),
                                                 ('$\\gamma=0.9142, \\beta=0.5$',np.sqrt(2)-1/2,1/2, (0, (3, 1, 1, 1, 1, 1))),]:

            for color, zeta_ in [('black',0), ('lightgray',0.1)]:
                if beta_ < round(1/4*(gamma_+1/2)**2,6): this_omega_bif = float(omega_bif.subs(zeta,zeta_).subs(gamma,gamma_).subs(beta,beta_))
                else: this_omega_bif = 10
                x = np.logspace(-2, np.log10(this_omega_bif),50)
                exp= sympy.plotting.experimental_lambdify.vectorized_lambdify([Omega],zeta_d.subs(eta, eta_).subs(gamma,gamma_).subs(beta,beta_).subs(zeta,zeta_))
                y = exp(x)#-zeta_
                #if zeta_: y/=zeta_

                #if method == 'Average Acceleration': label = "$\\zeta=\\SI{{{}}}{{\\percent}}_{{\mathrm{{crit}}}} $".format(zeta_*100)
                #else: label = None
                #print(label)

                #x/=np.sqrt(1-zeta_**2)# comparing undamped periods
                axes[1].semilogy( y, x/np.sqrt(1-zeta_**2)/2/np.pi,
                    label=label,
                    color = color, linestyle = linestyle)
        axes[1].set_xlabel('$\\hat{\\zeta} \; [\\si{\\percent}_{\mathrm{crit}}]$')
        #axes[1].set_ylabel('$\\nicefrac{h}{T}$')
        axes[1].set_xlim((-0.025,0.35))
        #axes[1].legend()
        plt.figlegend(loc=(0.66875,0.14))#.draggable()
        plt.subplots_adjust(left=0.069, bottom=0.126, right=0.979, top=0.975, wspace=0.044)
        axes[1].set_ylim((0.002,1))

        #plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.show()


if True:
    A1=sympy.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
    A2=sympy.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
    A=A1.solve(A2)
    with matplotlib.rc_context(rc=print_context_dict):

        fig,axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)

        rho_inf = 0.9047

        for method, linestyle in [('HHT-$\\alpha$','solid'),
                                 ('WBZ-$\\alpha$', 'dashed'),
                                 ('G-$\\alpha$','dotted'),
                                 ('Newmark-$\\beta$','dashdot'),]:

            alpha_f_, alpha_m_, gamma_, beta_ = None,None,None,None
            if method == 'Newmark-$\\beta$':
                alpha_f_ = 0
                alpha_m_ = 0
                gamma_ = (3-rho_inf)/(2*rho_inf+2)
                beta_ = 1/((rho_inf+1)**2)
            elif method == 'HHT-$\\alpha$':
                alpha_f_ = (1-rho_inf)/(rho_inf+1)
                alpha_m_ = 0
                gamma_ = 1/2+alpha_f_
                beta_ = (1+alpha_f_)**2/4
            elif method == 'WBZ-$\\alpha$':
                alpha_f_ = 0
                alpha_m_ = (rho_inf-1)/(rho_inf+1)
                gamma_ = 1/2-alpha_m_
                beta_ = (1-alpha_m_)**2/4
            elif method == 'G-$\\alpha$':
                alpha_m_ = (2*rho_inf-1)/(rho_inf+1)
                alpha_f_ = rho_inf/(rho_inf+1)
                gamma_ = 1/2-alpha_m_+alpha_f_
                beta_ = 1/4*(1/2+gamma_)**2
            
            print(alpha_f_, alpha_m_, gamma_, beta_)

            for color, zeta_ in [('black',0), ('lightgray',0.1)]:

                x = np.logspace(-2, 1)
                omega_d = np.empty_like(x)
                omega_d[:]=np.nan
                zeta_d = np.empty_like(x)
                zeta_d[:]=np.nan

                A_sub = A.subs(beta, beta_).subs(gamma, gamma_).subs(alpha_f, alpha_f_).subs(alpha_m,alpha_m_).subs(zeta, zeta_)

                for i in range(x.shape[0]):
                    #print(i)
                    lamdas = np.linalg.eigvals(np.array(A_sub.subs(Omega, x[i])).astype(np.float64))
#                     if i==41: print(lamdas)
                    if np.logical_and(np.real(lamdas)<0, np.imag(lamdas)==0).all(): 
                        #print(lamdas)
                        continue
                    
                    loglamdas = np.log(lamdas)
                    
                    rho = np.abs(lamdas)
                    phi = np.abs(np.angle(lamdas))
                    j = np.argmax((phi < np.pi)*rho)
                    #print(lamdas, loglamdas)
                    this_loglamda = loglamdas[loglamdas.imag>0].max()
#                     omega_d[i] = np.imag(this_loglamda) # damped
                    omega_d[i] = np.abs(this_loglamda) # comparing undamped periods
#                     omega_d[i] = np.sqrt(phi[j]**2+np.log(rho[j])**2)
                    zeta_d[i] = -np.real(this_loglamda)/np.abs(this_loglamda)
#                     zeta_d[i] = -np.log(rho[j])/omega_d[i]

                if zeta_ == 0: label = method
                else: label = None

                x/=np.sqrt(1-zeta_**2)# comparing undamped periods
                axes[0].semilogy((x/omega_d-1)*100, x/2/np.pi,
                    label=label,
                    color = color, linestyle = linestyle)

                y = zeta_d#-zeta_

                axes[1].semilogy( y, x/2/np.pi,
                    color = color, linestyle = linestyle)

        axes[0].set_xlabel('$\\sfrac{(\\hat{T} - T)}{T} \; [\\si{\\percent}]$')
        axes[0].set_ylabel('$\\sfrac{h}{T}$', labelpad=-15)
        axes[0].set_xlim((-35 ,170))
        axes[1].set_xlabel('$\\hat{\\zeta} \; [\\si{\\percent}_{\mathrm{crit}}]$')
        axes[1].set_xlim((-0.025,0.35))
        plt.figlegend(loc=(0.765,0.14))#.draggable()
        plt.subplots_adjust(left=0.069, bottom=0.126, right=0.979, top=0.975, wspace=0.044)
        axes[1].set_ylim((0.002,1))

        plt.show()


A1=sympy.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
A2=sympy.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
A=A1.solve(A2)
A = A.subs(gamma, 1/2-alpha_m+alpha_f).subs(beta, 1/4*(1-alpha_m+alpha_f)**2).subs(zeta, 0)

Aoo = sympy.Matrix([[0,0,0],[0,0,0],[0,0,0]])

for i in range(3):
    for j in range(3):
        Aoo[i,j]=sympy.limit(A[i,j],Omega,sympy.oo)

X = np.linspace(-10, 10, 21, endpoint=True)
Y = np.linspace(-10, 10, 21, endpoint=True)
rho_less = np.zeros((X.shape[0],Y.shape[0]))
Aoo = sympy.Matrix([[0,0,0],[0,0,0],[0,0,0]])

for i in range(3):
    for j in range(3):
        Aoo[i,j]=sympy.limit(A[i,j],Omega,sympy.oo)

eigvals = list(Aoo.eigenvals())
rho=sympy.Max(*[sympy.Abs(eigval) for eigval in eigvals])

p1 = sympy.plotting.plot_implicit(rho<1, (alpha_m,-1.1,1.1),(alpha_f,-1.1,1.1), line_color='grey',adaptive=False,show=False)
p2 = sympy.plotting.plot(sympy.solve(eigvals[0]-eigvals[1], alpha_f)[0], (alpha_m, -1,0.5), show=False, line_color='black')
p3 = sympy.plotting.plot(0.5, (alpha_m, -1.1,0.5), show=False, line_color='darkgray')
p4 = sympy.plotting.plot(alpha_m, (alpha_m, -1.1,0.5), show=False, line_color='darkgray')
p5 = sympy.plotting.plot_parametric(0,alpha_f, (alpha_f, 0,1/3), show=False, line_color='black')
p6 = sympy.plotting.plot(0, (alpha_m, -1,0), show=False, line_color='black')
p1.append(p2[0])
p1.append(p3[0])
p1.append(p4[0])
p1.append(p5[0])
p1.append(p6[0])
p1._backend = p1.backend(p1)
p1._backend.process_series()
ax = p1._backend.ax[0]
ax.annotate('$\\rho_\\infty \leq 1$', (-0.5,0.3), (-0.4,0.7), arrowprops={'arrowstyle':'->'})
ax.annotate('G-$\\alpha$', (0.25,0.417), (0.12,0.7), arrowprops={'arrowstyle':'->'})
ax.annotate('HHT-$\\alpha$', (0,0.17), (0.3,0.15), arrowprops={'arrowstyle':'->'})
ax.annotate('WBZ-$\\alpha$', (-0.4,0), (-0.3,-0.4), arrowprops={'arrowstyle':'->'})
plt.show()


X = np.linspace(-1.1, 1.1, 201, endpoint=True)
Y = np.linspace(-1.1, 1.1, 201, endpoint=True)
rho_mat = np.zeros((X.shape[0],Y.shape[0]))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        alpha_m_ = X[i]
        alpha_f_ = Y[j]
        rho3 = np.abs(alpha_f_/(alpha_f_-1))
        rho1 = np.abs((alpha_f_- alpha_m_ -1)/(alpha_f_-alpha_m_ + 1))
        rho_mat[j,i] = max(rho3, rho1)
rho_mat[rho_mat>1]=np.nan
plt.imshow(rho_mat, origin='lower', extent=[-1.1,1.1,-1.1,1.1])
plt.colorbar()
ax = plt.gca()
ax.plot([-1,0.5],[0,0.5], color='black')
ax.plot([-1,0],[0,0], color='black')
ax.plot([0,0],[0,1/3], color='black')
ax.annotate('G-$\\alpha$', (0.25,0.417), (0.12,0.7), arrowprops={'arrowstyle':'->'})
ax.annotate('HHT-$\\alpha$', (0,0.17), (0.3,0.15), arrowprops={'arrowstyle':'->'})
ax.annotate('WBZ-$\\alpha$', (-0.4,0), (-0.3,-0.4), arrowprops={'arrowstyle':'->'})

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlabel('$\\alpha_m$')
ax.set_ylabel('$\\alpha_f$')
plt.show()
