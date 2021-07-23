from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg") # force Matplotlib backend to Agg
plt.style.use('souza')
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.linewidth'] = 1
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# import PyJAGS
import pyjags

# import model and data
from createdata import *

# Create model code
line_code_jags = '''
model {{    
    beta1 ~ dnorm({mmu}, {minvvar})     # Gaussian prior on beta1
    beta0 ~ dnorm({mmu}, {minvvar})   # Gaussian prior on beta0
    log_sigma ~ dunif(-10, 10)
    sigma <- exp(log_sigma) 
    
    for (i in 1:N) {{
        y[i] ~ dnorm(beta0 + beta1 * x[i], 1/sigma**2) # Gaussian likelihood
    }}
    
    y_new ~ dnorm(beta0 + beta1*90, 1/sigma**2)
}}
'''

datadict = {'x': x,    # abscissa points (converted from numpy array to a list)
            'N': M,    # number of data points
            'y': data} # the observed data

Nsamples = 1000 # set the number of iterations of the sampler
chains = 4      # set the number of chains to run with

# dictionary for inputs into line_code
 
linedict = {}
linedict['mmu'] = 0.0           # mean of Gaussian prior distribution for beta1
linedict['minvvar'] = 1/1000**2   # inverse variance of Gaussian prior distribution for beta1
linedict['clower'] = -1000        # lower bound on uniform prior distribution for beta0
linedict['cupper'] = 1000         # upper bound on uniform prior distribution for beta0
#linedict['invvar'] = 1/sigma**2 # inverse variance of the data

# compile model
model = pyjags.Model(line_code_jags.format(**linedict), data=datadict, chains=chains)
samples = model.sample(Nsamples, vars=['beta1', 'beta0', 'y_new']) # perform sampling

b1chainjags = samples['beta1'].flatten()
b0chainjags = samples['beta0'].flatten()
yn = samples['y_new'].flatten()
 
# extract the samples
postsamples = np.vstack((b1chainjags, b0chainjags)).T
 
# plot posterior samples (if corner.py is installed)
try:
    import corner # import corner.py
except ImportError:
    sys.exit(1)

print('Number of posterior samples is {}'.format(postsamples.shape[0]))
 
fig = corner.corner(postsamples, labels=[r"$\beta_1$", r"$\beta_0$"], color='r', smooth=1, 
                        levels=(0.6,0.96),plot_density=0, fill_contours=10, plot_datapoints=1,
                        truths=[beta1d, beta0d],  show_titles=True, title_kwargs={"fontsize": 13})

 
fig.savefig('corner.pdf',bbox_inches='tight')
 
##### data fig 
fig, (ax) = plt.subplots(figsize=(7,5))
plt.scatter( x,data, color='r', s=35)
ax.set_xlabel(r'age',fontsize='20') 
ax.set_ylabel(r'distance',fontsize='20')
ax.set_xlim(0,100)
ax.set_ylim(0,700)
fig.savefig('data.pdf',bbox_inches='tight')

 
##### b0 x b1 fig 
fig, (ax) = plt.subplots(figsize=(7,5))
plt.scatter(b0chainjags,b1chainjags, color='g', s=12, alpha=0.2   )
ax.set_xlabel(r"$\beta_0$",fontsize='20') 
ax.set_ylabel(r"$\beta_1$",fontsize='20')
ax.set_xlim(480,650)
ax.set_ylim(-6,-1)
fig.savefig('b1xb0.pdf',bbox_inches='tight')

 
##### dist x age
fig, (ax) = plt.subplots(figsize=(7,5)) 
nn=40
xx=range(1, 101)
for i in range(nn) :
 plt.plot(xx, b1chainjags[i]*xx+b0chainjags[i], linewidth=1 )
plt.scatter( x,data, color='r', s=35)
ax.set_xlabel(r'age',fontsize='20') 
ax.set_ylabel(r'distance',fontsize='20')
ax.set_xlim(0,100)
ax.set_ylim(0,700)
fig.savefig('data_lines.pdf',bbox_inches='tight')


##############################
############################## 
 
 
fig, (ax) = plt.subplots(figsize=(7,5)) 
plt.plot(yn ,  linewidth=1 )
ax.set_xlabel(r'Nsample',fontsize='20') 
ax.set_ylabel(r'distance$(90)$',fontsize='20')
 
fig.savefig('ynew.pdf',bbox_inches='tight')

fig, (ax) = plt.subplots(figsize=(7,5))   
ax.set_xlabel(r'distance$(90)$ ',fontsize='20') 
ax.set_ylabel(r'$N$',fontsize='20')
 
ax.hist(yn)
fig.suptitle(r'mean = %s \ \ \ \ \ std = %s' %(("{:.1f}".format(np.mean(yn))),("{:.1f}".format(np.std(yn)))), fontsize=22)
 
fig.savefig('ynew_hist.pdf',bbox_inches='tight')