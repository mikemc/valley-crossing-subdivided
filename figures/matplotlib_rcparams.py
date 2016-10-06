# Configure matplotlib
# See this link for more info
# http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-
# reproducible-plots-with-matplotlib.html
from matplotlib import rcParams
# Font sizes
rcParams['font.size'] = 9
# rcParams['axes.labelsize'] = 'medium'
# rcParams['xtick.labelsize'] = 'small'
# rcParams['ytick.labelsize'] = 'small'
# rcParams['legend.fontsize'] = 'medium'
rcParams['axes.labelsize'] = '9'
rcParams['xtick.labelsize'] = '9'
rcParams['ytick.labelsize'] = '9'
rcParams['legend.fontsize'] = '9'
# Render text with Latex
rcParams['text.usetex'] = True

# Font to match Genetics style for figures, sans-serif
rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Computer Modern Roman']
rcParams['font.sans-serif'] = ['Latin Modern']
# The overleaf template uses Helvetica for sans serif text
# rcParams['font.sans-serif'] = ['Helvetica']

# Latex preamble; max one package per line
preamble = r"\usepackage{amsfonts}" \
+ r",\usepackage{amssymb}" \
+ r",\usepackage{amsmath}" \
+ r",\usepackage[T1]{fontenc}" \
+ r",\linespread{0.84}" \
+ r",\DeclareMathOperator{\E}{E}" \
+ r",\newcommand{\Fst}{{F_\text{ST}}}" \
+ r"\newcommand{\avg}[1]{\left< #1 \right>}"
rcParams['text.latex.preamble'] = preamble

# "scaled" option to helvet package causes option clash
# + r",\usepackage{helvet}" \
# + r",\usepackage[scaled=0.92]{helvet}" \ # Scaled clashes with linespread
# + r",\linespread{0.84}" \
# + r",\setlength{\parindent}{0pt}"

# Line and marker sizes
rcParams['lines.linewidth'] = 1
rcParams['lines.markersize'] = 3
rcParams['axes.linewidth'] = 0.5

## Ticks sizes. 
# Major
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
# Minor
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5

# Add padding between the axes and tick labels
rcParams['xtick.major.pad'] = 4
rcParams['xtick.minor.pad'] = 4
rcParams['ytick.major.pad'] = 3
rcParams['ytick.minor.pad'] = 3

