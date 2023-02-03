def get_pcd():
    print_context_dict = {'text.usetex':True,
                     'text.latex.preamble':r"\usepackage{siunitx}\n\usepackage{xfrac}", # on preamble related errors, make sure to show plot within rc context-manager
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
    # figsize=(5.53,2.96)#beamer 16:9
    # figsize=(3.69,2.96)#beamer 16:9
    # plot.rc('axes.formatter',use_locale=True) #german months
    return print_context_dict