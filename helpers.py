def get_pcd(purpose='print'):
    print_context_dict = {'text.usetex':True,
                     'text.latex.preamble':r"\usepackage{siunitx}\usepackage{xfrac}", # on preamble related errors, make sure to show plot within rc context-manager
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
                     'figure.figsize':(5.906, 5.906 / 1.618),  # print #150 mm \columnwidth
                     # 'figure.figsize':(5.906/2,5.906/2/1.618),#print #150 mm \columnwidth
                     # 'figure.figsize':(5.53/2,2.96),#beamer
                     # 'figure.figsize':(5.53/2*2,2.96*2),#beamer
                     'figure.dpi':100}
    
    if purpose=='print':#150 mm \columnwidth
        print_context_dict['figure.figsize'] = (5.906, 5.906 / 1.618)
    elif purpose=='print_half':
        print_context_dict['figure.figsize'] = (5.906 / 2, 5.906 / 1.618)
    elif purpose=='beamer':
        print_context_dict['figure.figsize'] = (5.53/2,2.96)
    elif purpose=='beamer_half':
        print_context_dict['figure.figsize'] = (5.53/2,2.96)
        
    # figsize=(5.53,2.96)#beamer 16:9
    # figsize=(3.69,2.96)#beamer 16:9
    # plot.rc('axes.formatter',use_locale=True) #german months
    return print_context_dict

def test():
    import matplotlib
    import matplotlib.pyplot as plt
    with matplotlib.rc_context(get_pcd()):
    
        plt.figure()
        plt.plot([1,2,3],[1,4,1])
        plt.xlabel('$\sfrac{x}{b}$')
        plt.ylabel('Nano [\si{\metre}]')
        plt.show()
        
        
    
    
    
if __name__ =='__main__':
    test()