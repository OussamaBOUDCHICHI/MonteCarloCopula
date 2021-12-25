# -*- coding : utf8 -*-
# author : BOUDCHICHI Oussama
# Copulas methods package based on R's package : VineCopula

from rpy2.robjects.packages import importr, data
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects.numpy2ri as rpyn
import warnings
warnings.filterwarnings('ignore')
import scipy.integrate as integrate
from scipy.special import binom
from scipy.interpolate import interp1d

VineCopula= importr('VineCopula')
MASS = importr('MASS')
base = importr('base')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
ro.r(""" getvals <- function(kde) {retour = data.frame(x = kde$x, y= kde$y, z = kde$z)
return(retour)}""")
getvals = ro.globalenv['getvals']



# % 
def plotKDE(data, theoritical_dist, title):
    data.plot(kind='kde', color='darkblue', label='Empirical density of log-returns')
    plt.yticks(plt.yticks()[0],[str(x)+'%' for x in plt.yticks()[0]])
    x = np.linspace(data.min(), data.max(), len(data))
    dist1 = theoritical_dist[0]; dist2 = theoritical_dist[0]
    
    DIST1 = getattr(stats, dist1)
    DIST2 = getattr(stats, dist2)

    PARAMS1 = DIST1.fit(data)
    PARAMS2 = DIST2.fit(data)
    
    plt.plot(x, DIST1.pdf(x, *PARAMS1), color='darkred', label = dist1 + ' Density')
    plt.plot(x, DIST2.pdf(x, *PARAMS2), color='black', linestyle = 'dashed', label = dist2 + ' Density')
    
    plt.legend()
    plt.title(title + ' empirical density', fontdict={'weight':'bold', 'size': 20})
    return None

# % 

def plotJoint(data, x, y):
    g = sns.jointplot(data = data, x=x, y=y, kind="reg",
    scatter_kws={'color':'darkred', 'edgecolor':'black'},
    marginal_kws={'color':'darkblue', 'edgecolor':'black'},
    height = 8
    )
    regline = g.ax_joint.get_lines()[0]
    regline.set_color('black')
    regline.set_zorder(5)
    r, _ = stats.pearsonr(data[x], data[y])
    g.ax_joint.legend([r'$\rho_{X,Y}$ = %.3f' %r])
    return None

# % 
def JointECDF(X, Y, x, y):
    if X.shape[0] != Y.shape[0]:
        print('Please enter equal size vectors, returning 0 ...')
        return 0.
    
    
    N = X.shape[0]
    values = np.array([sum((X <= i) & (Y <= j)) / N for i, j in zip(x, y)])
    return values
# % 
def sgn(X):
    return np.array([(x >= 0) * 1 - 1 * (x < 0) for x in X])

# % 
def ChiPlot(X, Y):
    F = ECDF(X); G = ECDF(Y)
    N = X.shape[0]
    F_i = (N * F(X) - 1) / (N - 1)
    G_i = (N * G(Y) - 1) / (N - 1)
    H_i = (N * JointECDF(X, Y, X, Y) - 1) / (N - 1)

    xhi = (H_i - F_i * G_i) / np.sqrt(F_i * (1 - F_i) * G_i * (1 - G_i))
    lamb = 4 * sgn((F_i - 0.5) * (G_i - 0.5)) * np.maximum((F_i - 0.5)**2, (G_i - 0.5)**2)
    plt.scatter(lamb, xhi, facecolors='darkred', edgecolors='black')
    plt.axhline(y = 0., linestyle="dashed", color='darkblue')
    plt.axhline(y = 1.54 / np.sqrt(N), linestyle="dashed", color='darkgreen')
    plt.axhline(y = -1.54 / np.sqrt(N), linestyle="dashed", color='darkgreen')
    plt.axvline(-4 * (1 / (N - 1) - 0.5 )**2, linestyle="dashed", color='steelblue')
    plt.axvline(4 * (1 / (N - 1) - 0.5 )**2, linestyle='dashed', color='steelblue')
    plt.title('Chi-plot', fontdict={'weight':'bold', 'size':20})
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\chi$')

    return None

# % Kendall's Plot
def Kplot(X, Y, interp = False, interval = None, nobs = None):

    def integrand(w, i, N):
        K = w * (1 - np.log(w))
        k = - np.log(w)
        return  N * binom(N - 1, i - 1) * w * k * (K ** (i - 1)) * ((1 - K) ** (N - i))
    
    N = X.shape[0]
    H_i = (N * JointECDF(X, Y, X, Y) - 1) / (N - 1)
    H_i.sort()
    W = np.array([ integrate.quad(lambda w : integrand(w, i , N), 0, 1)[0] for i in np.arange(1, N + 1)])

    def k(w):
        return w * (1 - np.log(w))

    plt.scatter(W, H_i, facecolor='darkred', edgecolors='black')
    plt.plot(np.linspace(0, 1, 100), k(np.linspace(0, 1, 100)), linestyle='dashed', color='black')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='darkblue', linestyle='dashed')
    if interp:
        f = interp1d(W, H_i)
        plt.scatter(np.linspace(interval[0], interval[1], nobs), f(np.linspace(interval[0], interval[1], nobs)),  facecolor='darkred', edgecolors='black')

    return None

# % Rank-Rank plot

def RankRankPlot(X, Y):
    F = ECDF(X); G = ECDF(Y)
    N = X.shape[0]
    F_i = (N * F(X) - 1) / (N - 1)
    G_i = (N * G(Y) - 1) / (N - 1)

    plt.scatter(F_i, G_i, facecolor='darkred', edgecolors='black')
    plt.xlabel(r'$(F_i)_{1\leq i\leq n}$', fontdict={'weight':'bold'})
    plt.ylabel(r'$(G_i)_{1\leq i\leq n}$', fontdict={'weight':'bold'})

    return None


# % Empirical dist fitting :
def dist_fit(data, alpha = 0.05):
    DIST_NAMES = ['t', 'nct', 'skewnorm', 'norminvgauss', 'cauchy']
    print('Candidates : ', DIST_NAMES, ' .....')
    def AIC(data, k, dist, params):
        DIST=getattr(stats, dist)
        logLik = np.sum(DIST.logpdf(data, *params))
        return 2*k - 2*logLik
    results={}
    for dist in DIST_NAMES:
        DIST = getattr(stats, dist)
        PARAMS = DIST.fit(data)
        TEST_PVAL = stats.kstest(data, dist, args=PARAMS)[1]
        TEST_STAT = stats.kstest(data, dist, args=PARAMS)[0]
        if(TEST_PVAL >= alpha):
            aic=AIC(data, k=len(PARAMS), dist=dist, params=PARAMS)
            TUP = PARAMS + (TEST_PVAL,TEST_STAT, aic)
            results[dist]=TUP

    
    print('(PARAMS, p-value, AIC)')
    return results 

# % Joint ECDF
def JointECDF(X, Y, x, y):
    if X.shape[0] != Y.shape[0]:
        print('Please enter equal size vectors, returning 0 ...')
        return 0.
    
    
    N = X.shape[0]
    values = np.array([sum((X <= i) & (Y <= j)) / N for i, j in zip(x, y)])
    return values

#  % Joint EPDF

def JointEPDF(x, y):
    X_1 = FloatVector(x.values)
    X_2 = FloatVector(y.values)
    KD = MASS.kde2d(X_1, X_2)
    KD = getvals(KD)
    with localconverter(ro.default_converter + pandas2ri.converter):
        kdevals = ro.conversion.rpy2py(KD)
    
    return kdevals

def CopulaFit(x, y, set, indeptest = False, level = 0.05):
    X_1 = FloatVector(x.values)
    X_2 = FloatVector(y.values)
    pX = VineCopula.pobs(X_1)
    pY = VineCopula.pobs(X_2)
    familySet = FloatVector(np.array(set))
    fit = VineCopula.BiCopSelect(pX, pY, familyset = familySet, indeptest = indeptest, level = level)
    base.summary(fit);
    return fit

def Clayton(u, v, theta):
    return np.array([(s**(-theta) + t**(-theta) - 1) ** (-1 / theta) for s, t in zip(u, v)])

def Gumbel(u, v, theta):
    return np.array([np.exp(- ( (-np.log(s))**theta + (-np.log(t))**theta) ** (1 / theta)) for s, t in zip(u, v)])

def Frank(u, v, theta):
    return np.array([(-1 / theta) * np.log(1 + ((np.exp(-theta * s) - 1) * (np.exp(-theta * t) - 1)) / (np.exp(-theta) - 1)) for s, t in zip(u, v)])

def Deheuvels(x, y, u, v):
    F = ECDF(x); G = ECDF(y)
    F_i = F(x); G_i = G(y);
    N = x.shape[0]
    values = np.array([sum((F_i <= i ) & ( G_i <= j)) / N for i, j in zip(u, v)])
    return values

def GOFstatistic(X, Y, Copula, param, distance = 'C-V-M'):

    if X.shape[0] != Y.shape[0]:
        print('Please enter equal size vectors, returning 0 ...')
        return 0.
    
    if distance not in ['C-V-M', 'K-S']:
        print('Please enter a valid distance name, returning 0 .....')
        return 0.
    
    
    F = ECDF(X); G = ECDF(Y)
    N = X.shape[0]
    F_i = N * F(X) / (N + 1); G_i = N * G(Y) / (N + 1)

    if distance == 'C-V-M':
        return ((Deheuvels(X, Y, F_i, G_i) - eval(Copula + '(F_i, G_i,' + param +')'))**2).sum()
    if distance == 'K-S':
        return np.abs(Deheuvels(X, Y, F_i, G_i) - eval(Copula + '(F_i, G_i,' + param +')')).max()

def payoff(omega, price, K):
    return np.maximum((omega * price).sum() - K, 0.)

def CopulaMC(N, S_0, K, days, r, omega, copula, copPar, marginal, margPar):
    DIST = getattr(stats, marginal)
    price = 0

    for i in range(N):
        sim = VineCopula.BiCopSim(days, copula, par = copPar)

        with localconverter(ro.default_converter + pandas2ri.converter):
            Sim = ro.conversion.rpy2py(sim)
        Sim = pd.DataFrame(Sim)

        SimX=  DIST.ppf(Sim.iloc[:, 0], *margPar) / days
        SimY = stats.nct.ppf(Sim.iloc[:, 1], *margPar) / days

        priceX = S_0 * np.exp(SimX.sum())
        priceY = S_0 * np.exp(SimY.sum())

        inPrice = np.array([priceX, priceY])
        price += payoff(omega, inPrice, K)

    return np.exp(-r * (days/ 365)) * (price / N)