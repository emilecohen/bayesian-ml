import numpy as np
import numpy.random as npr
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from sklearn.linear_model import Lasso

def generate_data(sample_size, dimension, seed):
    """generate simulated data with controlled theta_true
    """
    npr.seed(seed)
    # Set up parameters
    intercept = 0 # I set the intercept to zero for simplicity. Otherwise,
                  # you'd have to include an intercept in your models.
    sigma_noise = 1
    proportion_of_nonzero_coefficients = 0.1
    signal_absolute_value = 10

    # Prepare containers to return
    theta_true = np.zeros(dimension)
    support = np.zeros(dimension)

    for i in range(dimension):
        if bernoulli(proportion_of_nonzero_coefficients).rvs():
            if bernoulli(0.5).rvs():
                theta_true[i] = np.random.normal(signal_absolute_value, 1)
            else:
                theta_true[i] = np.random.normal(-signal_absolute_value, 1)
            support[i] = 1

        else:
            theta_true[i] = np.random.normal(0, 0.25)

    X = np.random.normal(0, 1, (sample_size, dimension))
    y = np.random.normal(X.dot(theta_true) + intercept, sigma_noise)

    return X, y, theta_true, sigma_noise, np.where(support)[0]


def plot_coefficients(axes, theta_true, indices_support, theta_hat, lower_bound=None, upper_bound=None, color='b', label=""):
    """plot theta_true and estimated theta, with filled in error bars. Then plot residuals.
    """
    indices = range(len(theta_true))

    # Plot credible intervals
    axes[0].plot(indices, theta_true, 'g', alpha=0.5, linewidth=3, label="true values")
    axes[0].plot(indices, theta_hat, color=color, alpha=0.5, linewidth=3, label=label+" estimate")

    if lower_bound is not None:
        axes[0].fill_between(indices, upper_bound, lower_bound, color=color, alpha=0.3)

    ymin, ymax = axes[0].get_ylim()
    delta = ymax-ymin
    axes[0].vlines(indices_support, ymin-.1*delta, ymax+.1*delta, linestyle='--', color='g')
    axes[0].set_ylim([ymin, ymax])

    # Plot residuals
    axes[1].plot(indices, theta_hat-theta_true, color=color, alpha=0.5, linewidth=3, label=label+" residual")

    if lower_bound is not None:
        axes[1].fill_between(indices, upper_bound-theta_true, lower_bound-theta_true, color=color, alpha=0.3)

    ymin, ymax = axes[1].get_ylim()
    delta = ymax-ymin
    axes[1].vlines(indices_support, ymin-.1*delta, ymax+.1*delta, linestyle='--', color='g')
    axes[1].set_ylim([ymin, ymax])
    plt.legend()

    return # axes

def get_sklearn_lasso_estimate(X, y):
    """apply scikit-learn lasso. This should return an estimated theta_hat.
    """
    # here should go your code.
    clf = Lasso(alpha=0.1)
    clf.fit(X,y)
    
    return clf.coef_

def get_mcmc_sample_for_laplace_prior(X, y):
    # This should return a pymc3 Trace object
    with pm.Model() as laplace_model:
        theta = pm.Laplace("theta", mu=0, b=.5, shape=X.shape[1]) # mu and b are hyperparameters
        mu = tt.dot(X, theta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=1, observed=y)
        
        trace = pm.sample(500, return_inferencedata=False) # we choose to sample 500 points
        
    return trace

def get_mcmc_sample_for_horseshoe_prior(X, y):
    # This should return a pymc3 Trace object
    with pm.Model() as horseshoe_model:
        λ = pm.HalfCauchy('lambda', beta=1, shape=X.shape[1])
        τ = pm.HalfCauchy('tau', beta=1)
        σ = pm.Deterministic('horseshoe', τ*τ*λ*λ)
        theta = pm.Normal('theta', mu=0, sd=σ, shape=X.shape[1])

        mu = tt.dot(X, theta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=1, observed=y)
       
        trace = pm.sample(1000, return_inferencedata=False) # we choose to sample 500 points
        
    return trace

def get_mcmc_sample_for_finnish_horseshoe_prior(X, y):
    # This should return a pymc3 Trace object
    m0=10
    slab_scale = 3
    slab_scale_squared=slab_scale*slab_scale
    slab_degrees_of_freedom=25
    half_slab_df=slab_degrees_of_freedom*0.5

    with pm.Model() as finnish_horseshoe_prior:

        tau0 = (m0 / (X.shape[1] - m0)) * (1 / np.sqrt(1.0 * X.shape[0]))

        beta_tilde = pm.Normal('beta_tilde', mu=0, sd=1, shape=X.shape[1], testval=0.1)
        lamda = pm.HalfCauchy('lamda', beta=1, shape=X.shape[1], testval=1.0)
        tau_tilde = pm.HalfCauchy('tau_tilde', beta=1, testval=0.1)
        c2_tilde = pm.InverseGamma('c2_tilde', alpha=half_slab_df, beta=half_slab_df, testval=0.5)


        tau=pm.Deterministic('tau', tau_tilde*tau0)
        c2=pm.Deterministic('c2',slab_scale_squared*c2_tilde)
        lamda_tilde =pm.Deterministic('lamda_tilde', pm.math.sqrt((c2 * pm.math.sqr(lamda) / (c2 + pm.math.sqr(tau) * pm.math.sqr(lamda)) ))) 
        
        theta = pm.Deterministic('theta', tau * lamda_tilde * beta_tilde)
        mu = tt.dot(X, theta)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=1, observed=y)
      
        trace = pm.sample(1000, return_inferencedata=False) # we choose to sample 500 points
        
        
    return trace

if __name__ == '__main__':

    X, y, theta, support = generate_data(100, 200, 1)
