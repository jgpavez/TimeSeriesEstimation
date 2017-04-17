import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import theano
import scipy.linalg as L
import pdb

import os, sys, inspect
#cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
#if cmd_folder not in sys.path:
    
sys.path.insert(0,'/afs/cern.ch/user/j/jpavezse/systematics/carl')

import carl
from itertools import product


from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Dropout
from carl.learning import as_classifier
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from carl.ratios import ClassifierRatio
from carl.learning import CalibratedClassifierCV
from carl.learning import as_classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import multiprocessing

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata

import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.metrics import log_loss

import statsmodels.tsa.vector_ar.var_model as vector_ar

sys.setrecursionlimit(10000)

matplotlib.style.use('ggplot')

np.random.seed(1234)

p_value_ = 1
ndims_ = 2
nparams_ = 2
N_ = 20000
T_ = 50

LOAD_DATA = False
TRAIN = True

class VAR:

    def sample(self, coefs, intercept, sigma_2, n_steps, rng=None):
        return vector_ar.util.varsim(coefs, intercept, sigma_2, steps=n_steps)
    
    def plot(self,steps=1000):
        self.VAR.plotsim(steps)

    def nll(self, X, alphas, intercept, sigma_2):
        lags = 1
        trend = 'c'
        alpha_0 = alphas[0]
        alpha_1 = alphas[1]
        coefs = np.array([alpha_0,0.,0.,alpha_1]
                        ).reshape(p_value_,ndims_,ndims_)
        
        VAR = vector_ar.VAR(X)
        # have to do this again because select_order doesn't call fit
        VAR.k_trend = k_trend = vector_ar.util.get_trendorder(trend)

        offset = 0
        y = VAR.y[offset:]

        z = vector_ar.util.get_var_endog(y, lags, trend, 
                                         has_constant='skip')
        y_sample = y[lags:]
        intercept = intercept
        params = np.vstack((intercept, coefs.reshape((2,2))))
        #params = np.linalg.lstsq(z, y_sample)[0]
        resid = y_sample - np.dot(z, params)
        omega = np.array(sigma_2)
        
        lastterm = -np.trace(np.dot(np.dot(resid, L.inv(omega)),resid.T))
        
        varfit = vector_ar.VARResults(y, z, params, omega, lags, names=VAR.endog_names,
                    trend=trend, dates=VAR.data.dates, model=self)
        llf = -varfit.llf
        llf += 0.5*lastterm
        return -llf
    
    def rvs(self, n_samples, alpha_0, alpha_1, intercept, sigma_2, random_state=1234):
        rng = np.random.RandomState(random_state) if \
                isinstance(random_state,int) else random_state
        coefs = np.array([alpha_0,0.,0.,alpha_1]
                        ).reshape(p_value_,ndims_,ndims_)
        return self.sample(coefs=coefs,intercept=intercept,
                           sigma_2=sigma_2,
                           n_steps=n_samples,rng=rng)


alpha_value_0 = 0.5
alpha_value_1 = 0.4
#alpha_value = np.array([0.5,0.,0.5,0.]).reshape(1,ndims,ndims)
sigma_2_value = [[1.,0.],[0.,1.]]
intercept=[0.,0.]
alpha_value_0_bkg = 0.3
alpha_value_1_bkg = 0.3
#alpha_value_bkg = np.array([0.3,0.,0.3,0.]).reshape(1,ndims,ndims)
sigma_2_value_bkg = [[1.,0.],[0.,1.]]


bounds = [(0.3, 0.6), (0.3, 0.6)]
n_points = 5

As = np.linspace(bounds[0][0],bounds[0][1], n_points)
Bs = np.linspace(bounds[1][0],bounds[1][1], n_points)
AA, BB = np.meshgrid(As, Bs)
AABB = np.hstack((AA.reshape(-1, 1),
               BB.reshape(-1, 1)))

p0 = VAR()
#p1 = Ricker(r=np.exp(4.5), sigma_2=0.3**2,phi=10.)
p1 = VAR()
rng = np.random.RandomState(1234)

n_true = 1000
#X_true = p0.rvs(1000,random_state=rng)
X_true = np.array([p0.rvs(T_, alpha_value_0, alpha_value_1, intercept,
                          sigma_2_value, random_state=np.random.randint(0,5000))
                   for i in range(n_true)])

bounds_values = [(np.linspace(bounds[0][0],bounds[0][1], num=n_points)),
                 (np.linspace(bounds[1][0],bounds[1][1], num=n_points))]
combinations = list(product(*bounds_values))
  
print('Start producig data')
Xs = []
ys = []

def produce_data(value):
    return np.array([p0.rvs(T_, value[0], value[1], intercept, sigma_2_value,
                           random_state=np.random.randint(0,5000))
                   for i in range(N_//2)])   

num_cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(num_cores)

values = [v for _,v in enumerate(combinations)]

if LOAD_DATA:
    Xs = pool.map(produce_data, values)
    X1_ = np.array([p1.rvs(T_, alpha_value_0_bkg, alpha_value_1_bkg, 
                           intercept, sigma_2_value,
                           random_state=np.random.randint(0,5000))
                   for i in range(N_//2)])
    pickle.dump([Xs, X1_], open('data/ar_data.dat', 'w'))
    pool.close()
else:
    Xs, X1_ = pickle.load(open('data/ar_data.dat', 'r'))

print('End producing data')

Xs = np.array(Xs)

#Xs_min = Xs.min(axis=1).min(axis=0).min(axis=0)
#X1_min = X1_.min(axis=0).min(axis=0)
#X_true_min = X_true.min(axis=0).min(axis=0)
#Xs_min = np.vstack((Xs_min, X1_min, X_true_min)).min(axis=0)

#X_c = np.log(Xs - Xs_min + 1.)
#X_true_c = np.log(X_true - Xs_min + 1.)
#X1_c = np.log(X1_ - Xs_min + 1.)

#Xs = X_c
#X_true = X_true_c
#X1_ = X1_c


def make_model_join():
    model = Sequential()
    model.add(GRU(15, input_shape=(T_, ndims_)))
    model.add(Dense(5, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    sgd = Adam(clipnorm=40.)
    model.compile(loss="binary_crossentropy", optimizer='adam')
    return model


def make_ratio(num):
    X_num = Xs[num]
    X_den = X1_
    X = np.vstack((X_num, X_den))
    y = np.zeros(len(X_num) + len(X_den), dtype=np.int)
    y[len(X_num):] = 1

    clf = KerasClassifier(make_model_join, nb_epoch=50, verbose=0)

    cv =  StratifiedShuffleSplit(n_iter=1, test_size=0.5, random_state=1)

    ratio = ClassifierRatio(
        base_estimator=CalibratedClassifierCV(clf, cv=cv, bins=20),
        random_state=0)
    ratio.fit(X, y)
    
    print('Loss {0} : {1}'.format(num, log_loss(ratio.classifier_.classifiers_[0].
                   predict(X[:int(len(X)*0.3)]),y[:int(len(X)*0.3)])))
    
    return ratio


if TRAIN:
    print('Start training')
    clf_ratios = []

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    n_sample_points = len(combinations)
    points = list(range(n_sample_points))
    for i, k in enumerate([points[n:n + num_cores] for n in range(0, n_sample_points, num_cores)]):
        print('Iteration {0}'.format(i))
        clf_ratios += pool.map(make_ratio, k)
    pool.close()

    print('End training')

    llr = []
    zeros = []
    distances = []

    for i, theta in enumerate(combinations):
        print(i, theta)
        ratio = clf_ratios[i]
        ratios = ratio.predict(X_true, log=True)
        print ratios[np.isinf(ratios)].shape
        zeros.append(ratios[np.isinf(ratios)].shape[0])
        ratios = ratios[np.logical_and(np.isfinite(ratios),~np.isnan(ratios))]
        nllr = -np.mean(ratios) 
        llr.append(nllr)
        print(llr[-1])

    pickle.dump([llr, zeros], open('data/ar_results.dat','w'))
else:
    llr, zeros = pickle.load(open('data/ar_results.dat', 'r'))
    
llr = np.array(llr)
llr[np.isnan(llr)] = 0.
#llr *= 19.2 / len(X_true)

# Computing exact likelihood
exact_contours = np.zeros(len(AABB))
combinations = product(As, Bs)
i = 0
for a in As:    
    for b in Bs:
        exact_contours[i] = np.mean([p0.nll(X_true[k], [a,b], intercept, sigma_2_value) 
                                     for k in range(100)])
        i += 1
#exact_contours *= 19.2 / len(X_true)


plt.rcParams["figure.figsize"] = (12, 10)

thetas = np.array([v for v in product(As, Bs)])

llr_ = np.array(llr).reshape(n_points, n_points)
llr_ = np.flipud(llr_.transpose())
mle = np.unravel_index(llr_.argmin(),llr_.shape)
llr_ -= llr_[mle]
llr_ *= 2.
plt.imshow(llr_, aspect='auto', cmap='viridis',
           extent = (bounds[0][0], bounds[0][1],bounds[1][0], bounds[1][1]))

plt.colorbar()  

plt.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
plt.scatter([As[mle[1]]], [Bs[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
plt.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)

plt.savefig('plots/ar_results.pdf')

plt.clf()

zeros = np.array(zeros)
zeros = np.flipud(zeros.transpose())

plt.imshow(zeros.reshape(n_points,n_points), aspect='auto', cmap='viridis',
           extent = (bounds[0][0], bounds[0][1],bounds[1][0], bounds[1][1]))

plt.colorbar()  
plt.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
plt.scatter([As[mle[1]]], [Bs[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
plt.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)

plt.savefig('plots/ar_zeros.pdf')

plt.clf()


# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.1f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:
    fmt = r'%r '
else:
    fmt = '%r '

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
ax = axes.flat[0]

thetas = np.array([v for v in product(As, Bs)])

gp = GaussianProcessRegressor(alpha=0.0, normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
#gp.fit(np.delete(theta[:29, 1:],8,0), np.delete(llr,8))
gp.fit(thetas, llr)

xi = np.linspace(bounds[0][0], bounds[0][1], 500)
yi = np.linspace(bounds[1][0], bounds[1][1], 500)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs = ax.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
cs.levels = [nf(val) for val in cs.levels]
ax.clabel(cs, cs.levels, inline=1, fontsize=10, fmt = fmt)
cs = ax.contourf(xi, yi, zi, 30, cmap="viridis",
                  vmax=abs(zi).max(), vmin=0.0)

#plt.colorbar()
ax.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
ax.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
ax.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)
#plt.scatter([theta[theta1, 1]], [theta[theta1, 2]], marker='o', c='r', s=50, lw=0, zorder=10)

ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])
ax.set_xlabel(r"$\alpha_0$", size=16)
ax.set_ylabel(r"$\alpha_1$", size=16)
ax.set_title("Approx. -2lnL (VAR(2,1))")

ax = axes.flat[1]

thetas = np.array([c for c in product(As, Bs)])
gp = GaussianProcessRegressor(alpha=0.0, normalize_y=True, 
                              kernel=C(1.0) * Matern(1.0, length_scale_bounds="fixed"))
#gp.fit(np.delete(theta[:29, 1:],8,0), np.delete(llr,8))
gp.fit(thetas, exact_contours)

xi = np.linspace(bounds[0][0], bounds[0][1], 500)
yi = np.linspace(bounds[1][0], bounds[1][1], 500)
    
xx, yy = np.meshgrid(xi, yi)
zz, std = gp.predict(np.c_[xx.ravel(), yy.ravel()], return_std=True)
zi = zz.reshape(xx.shape)

mle = np.unravel_index(zi.argmin(),zi.shape)
zi -= zi[mle]
zi *= 2.

cs2 = ax.contour(xi, yi, zi, [1.,4.,9.,16.,25.], linewidths=0.5, colors='w')
cs2.levels = [nf(val) for val in cs2.levels]
ax.clabel(cs2, cs2.levels, inline=1, fontsize=10, fmt = fmt)
cs2 = ax.contourf(xi, yi, zi, 30, cmap="viridis",
                  vmax=abs(zi).max(), vmin=0.0)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.81, 0.12, 0.03, 0.775])
fig.colorbar(cs2, cax=cbar_ax)

ax.scatter(thetas[:,0],thetas[:,1], marker='o', c='b', s=50, lw=0, zorder=10)
smle = ax.scatter([xi[mle[1]]], [yi[mle[0]]], marker='o', c='r', s=50, lw=0, zorder=10)
sobs = ax.scatter([alpha_value_0],[alpha_value_1], marker='o', c='w', s=50, lw=0, zorder=10)

lines = [smle, sobs]
labels = ['MLE', 'Observed']

ax.legend(lines, labels, frameon=False, prop={'size':12}, scatterpoints=1)

ax.set_xlim(bounds[0][0], bounds[0][1])
ax.set_ylim(bounds[1][0], bounds[1][1])
ax.set_xlabel(r"$\alpha_0$", size=16)
ax.set_ylabel(r"$\alpha_1$", size=16)
ax.set_title("Exact. -2lnL (VAR(2,1))")

plt.savefig('plots/ar_gp.pdf')


plt.close()
plt.clf()
