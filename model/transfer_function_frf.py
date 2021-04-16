
# coding: utf-8

# In[101]:


#%matplotlib notebook
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
print_context_dict ={'text.usetex':True,
                     'text.latex.preamble':r"\usepackage{siunitx},\usepackage{nicefrac}",
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
                     'figure.dpi':100}
    #figsize=(5.53,2.96)#beamer 16:9
    #figsize=(3.69,2.96)#beamer 16:9
    #plot.rc('axes.formatter',use_locale=True) #german months
# must be manually set due to some matplotlib bugs
if print_context_dict['text.usetex']:
    plt.rc('text.latex',unicode=True)
    plt.rc('text',usetex=True)


from sympy import *
import sympy.plotting.experimental_lambdify
init_printing()

f,k,d,m,zeta,fn = symbols('\omega k d m \zeta \omega_n')
H = (1/k)/(1-(f/fn)**2+1j*2*zeta*(f/fn))
abs(H.subs(f,fn))


# In[107]:
@ticker.FuncFormatter
def major_formatter(x, pos):
    x=('%.15f' % x).rstrip('0').rstrip('.')
    return f'$\\frac{{{x}}}{{k}}$'

with matplotlib.rc_context(rc=print_context_dict):
    fig,[ax1,ax2]=plt.subplots(2,1,gridspec_kw={'height_ratios':[3,1]}, sharex=True)
    for linestyle, zeta_ in [('solid',0), ('dashed',0.01), ('dotted',0.1), ('dashdot',1)]:#, ((0, (3, 1, 1, 1, 1, 1)),1)]:
        x = np.logspace(-1,1,10000)
       
        #exp= sympy.plotting.experimental_lambdify.vectorized_lambdify([f],(abs(H.subs(fn,1).subs(k,1).subs(zeta,zeta_))))
        label=f'$\\zeta = {zeta_}$'
        y = abs(1/(1*(-(x**2/1**2)+2*1j/1*x*zeta_+1)))
        ax1.loglog(x,y,
            label=label,
            linestyle = linestyle, color='black', alpha=0.5)
        #exp = sympy.plotting.experimental_lambdify.vectorized_lambdify([f],(arg(H.subs(fn,10).subs(k,1).subs(zeta,zeta_))))
        
        y = -np.angle(1/(1*(-(x**2/1**2)+2*1j/1*x*zeta_+1)))/np.pi*180
        ax2.plot(x,y,linestyle=linestyle, color='black', alpha=0.5)
    # for zeta_ in [0.001,0.1,0.5,1]:
    #     p=plot(abs(H.subs(fn,10).subs(k,1).subs(zeta,zeta_)), (f,0.1,100),xscale='log',yscale='log', show=false)
    #     ps.append(p)
    # for p in ps[1:]:
    #     ps[0].append(p[0])
    # ps[0].show()
    ax2.set_xlabel('\Large $\\nicefrac{\omega}{\omega_n}$')
    ax1.set_ylabel('$|H(\omega)_{f-d}|$  [\si{\metre}]')
    ax2.set_ylabel('$\\arg\\bigl(H(\omega)_{f-d}\\bigr)$ [\si{\degree}]')
    ax1.yaxis.set_major_formatter(major_formatter)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(90))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(45))
    
    ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
    ax2.set_xlim((0.1,10))
    #axes[1].legend()
    plt.figlegend(loc=(0.78,0.75)).draggable()
    plt.subplots_adjust(left=0.109, bottom=0.131, right=0.979, top=0.975, wspace=0.044)
    #ax1.set_ylim((0,))

    #plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


# In[76]:



plt.ion()
plt.figure()
ax1 = plt.subplot()
plt.figure()
ax2 = plt.subplot()
for exponent in range(4,8):
    noise = (np.random.random(10**exponent)-0.5)*4
    #noise = (np.random.uniform(size=10**exponent))
    t=np.linspace(0,1,10**exponent)
    ax1.plot(t,noise, alpha=0.1)
    sampling_rate = 1/(t[1]-t[0])
    n_lines=1000
    _, Pxy_den = scipy.signal.csd(noise,noise, 
                                  sampling_rate, nperseg=2*n_lines, scaling='spectrum', return_onesided=True)    
    Pxy_den *= n_lines
    freqs = np.fft.rfftfreq(2*n_lines , 1/sampling_rate) 
    
    ax2.plot(freqs,Pxy_den)


# In[ ]:




