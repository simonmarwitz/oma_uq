import matplotlib
import re

# usage
'''
# plot may be generated prior or within rc_context

from helpers import get_pcd
pcd = get_pcd('print')
with plt.rc_context(pcd):
    plt.show()
    plt.savefig('/home/sima9999/test.pdf', backend='pgf')
'''


def get_pcd(purpose='print'):
    # https://github.com/matplotlib/matplotlib/issues/17931
    matplotlib.rcParams.update({
#     'text.usetex':True,
#     'pgf.texsystem':'lualatex',
#     'pgf.rcfonts': False,
#     "pgf.preamble": r"\usepackage{siunitx}\usepackage{xfrac}\usepackage{unicode-math}\setmathfont{latinmodern-math.otf}\setmathfont[range=\mathfrak]{Old English Text MT}",
    'text.latex.preamble':r"\usepackage{siunitx}\usepackage{xfrac}\usepackage{amssymb, amsfonts, amsmath}\usepackage{eufrak}",  # on preamble related errors, make sure to show plot within rc context-manager

    })
    print_context_dict = {'text.usetex':True,  # for interactive plotting
                    # 'text.latex.preamble':r"\usepackage{siunitx}\usepackage{xfrac}\usepackage{amssymb, amsfonts}\usepackage{eufrak}",  # on preamble related errors, make sure to show plot within rc context-manager
                          'pgf.texsystem': 'lualatex',  # for saving figures with lualatex, use ...savefig('.. .pdf', backend='pgf')
                          'pgf.rcfonts': False,
                          'pgf.preamble': r"\usepackage{siunitx}\usepackage{xfrac}\usepackage{amsmath}\usepackage{unicode-math}\setmathfont{latinmodern-math.otf}\setmathfont[range=\mathfrak]{Old English Text MT}\setmathfont[range=\mathbb]{texgyrepagella-math.otf}",
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

    if purpose == 'print':  # 150 mm \columnwidth
        print_context_dict['figure.figsize'] = (5.906, 5.906 / 1.618)
    elif purpose == 'print_half':
        print_context_dict['figure.figsize'] = (5.906 / 2, 5.906 / 1.618)
    elif purpose == 'print_half_hor':
        print_context_dict['figure.figsize'] = (5.906 , 5.906 / 1.618 / 2)
    elif purpose == 'beamer':
        print_context_dict['figure.figsize'] = (5.53, 2.96)
    elif purpose == 'beamer_half':
        print_context_dict['figure.figsize'] = (5.53 / 2, 2.96)
        print_context_dict['font.size'] = 9
        print_context_dict['legend.fontsize'] = 9
        print_context_dict['xtick.labelsize'] = 9
        print_context_dict['ytick.labelsize'] = 9
        print_context_dict['axes.labelsize'] = 9
    elif purpose == 'BB15':
        # print_context_dict['figure.figsize'] = (130/25.4, 44/25.4)   # schmale version
        print_context_dict['figure.figsize'] = (511 / 100, 316 / 100)
        print_context_dict['font.size'] = 9
        print_context_dict['legend.fontsize'] = 9
        print_context_dict['xtick.labelsize'] = 9
        print_context_dict['ytick.labelsize'] = 9
        print_context_dict['axes.labelsize'] = 9

    # figsize=(5.53,2.96)#beamer 16:9
    # figsize=(3.69,2.96)#beamer 16:9
    # plot.rc('axes.formatter',use_locale=True) #german months
    return print_context_dict


def test():
    import matplotlib.pyplot as plt
    pcd = get_pcd('print')
    with matplotlib.rc_context(pcd):
        plt.figure()
        plt.title(r'$n_\mathrm{ord} \mathfrak{k} \mathcal{h}$ \si{\metre\per\second\squared}')
        # plt.title(r'$\si{\metre}$')
        plt.savefig('/home/sima9999/test.pdf', backend='pgf')
        plt.show()
        print('\nInside context:')
        print(f"  'text.usetex': {mpl.rcParams['text.usetex']}")
        print(f"  'text.latex.preamble': {mpl.rcParams['text.latex.preamble']}")


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item:-len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


if __name__ == '__main__':
    test()
