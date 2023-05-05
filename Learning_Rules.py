import matplotlib.pyplot as plt
import numpy as np
import time

""" STDP """


def default_pars_STDP(**kwargs):
    pars = {}

    # typical neuron parameters
    pars['V_th'] = -55.  # spike threshold [mV]
    pars['V_reset'] = -75.  # reset potential [mV]
    pars['tau_m'] = 10.  # membrane time constant [ms]
    pars['V_init'] = -65.  # initial potential [mV]
    pars['V_L'] = -75.  # leak reversal potential [mV]
    pars['tref'] = 2.  # refractory time (ms)

    # STDP parameters
    pars['A_plus'] = 0.008  # magnitude of LTP
    pars['A_minus'] = pars['A_plus'] * 1.10  # magnitude of LTD
    pars['tau_stdp'] = 20.  # STDP time constant [ms]

    # simulation parameters
    pars['T'] = 400.  # Total duration of simulation [ms]
    pars['dt'] = .1  # Simulation time step [ms]

    # external parameters if any
    for k in kwargs:
        pars[k] = kwargs[k]

    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

    return pars


def my_raster_plot(range_t, spike_train, n):
    """Generates poisson trains

  Args:
    range_t     : time sequence
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of Poisson trains plot

  Returns:
    Raster_plot of the spike train
  """

    # Find the number of all the spike trains
    N = spike_train.shape[0]

    # n should be smaller than N:
    if n > N:
        print('The number n exceeds the size of spike trains')
        print('The number n is set to be the size of spike trains')
        n = N

    # Raster plot
    i = 0
    while i <= n:
        if spike_train[i, :].sum() > 0.:
            t_sp = range_t[spike_train[i, :] > 0.5]  # spike times
            plt.plot(t_sp, i * np.ones(len(t_sp)), 'k|', ms=10, markeredgewidth=2)
        i += 1
    plt.xlim([range_t[0], range_t[-1]])
    plt.ylim([-0.5, n + 0.5])
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')


def my_example_P(pre_spike_train_ex, pars, P):
    """Generates two plots (raster plot and LTP vs time plot)

  Args:
    pre_spike_train_ex     : spike-train
    pars : dictionary with the parameters
    P : LTP ratio

  Returns:
    my_example_P returns a rastert plot (top),
    and a LTP ratio across time (bottom)
  """
    spT = pre_spike_train_ex
    plt.figure(figsize=(7, 6))
    plt.subplot(211)
    color_set = ['red', 'blue', 'black', 'orange', 'cyan']
    for i in range(spT.shape[0]):
        t_sp = pars['range_t'][spT[i, :] > 0.5]  # spike times
        plt.plot(t_sp, i * np.ones(len(t_sp)), '|',
                 color=color_set[i],
                 ms=10, markeredgewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.xlim(0, 200)

    plt.subplot(212)
    for k in range(5):
        plt.plot(pars['range_t'], P[k, :], color=color_set[k], lw=1.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('P(t)')
    plt.xlim(0, 200)

    plt.tight_layout()
    plt.show()


def mySTDP_plot(A_plus, A_minus, tau_stdp, time_diff, dW):
    plt.figure()
    plt.plot([-5 * tau_stdp, 5 * tau_stdp], [0, 0], 'k', linestyle=':')
    plt.plot([0, 0], [-A_minus, A_plus], 'k', linestyle=':')

    plt.plot(time_diff[time_diff <= 0], dW[time_diff <= 0], 'ro')
    plt.plot(time_diff[time_diff > 0], dW[time_diff > 0], 'bo')

    plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
    plt.ylabel(r'$\Delta$W', fontsize=12)
    plt.title(' STDP', fontsize=12, fontweight='bold')
    plt.show()


def Delta_W(pars, A_plus, A_minus, tau_stdp):
    """
  Plot STDP biphasic exponential decaying function
  Args:
    pars       : parameter dictionary
    A_plus     : (float) maximum amount of synaptic modification
                 which occurs when the timing difference between pre- and
                 post-synaptic spikes is positive
    A_minus    : (float) maximum amount of synaptic modification
                 which occurs when the timing difference between pre- and
                 post-synaptic spikes is negative
    tau_stdp   : the ranges of pre-to-postsynaptic interspike intervals
                 over which synaptic strengthening or weakening occurs
  Returns:
    dW         : instantaneous change in weights
  """

    # STDP change
    dW = np.zeros(len(time_diff))
    # Calculate dW for LTP
    dW[time_diff <= 0] = A_plus * np.exp(time_diff[time_diff <= 0] / tau_stdp)
    # Calculate dW for LTD
    dW[time_diff > 0] = -A_minus * np.exp(-time_diff[time_diff > 0] / tau_stdp)

    return dW


pars = default_pars_STDP()
# Get parameters
A_plus, A_minus, tau_stdp = pars['A_plus'], pars['A_minus'], pars['tau_stdp']
# pre_spike time - post_spike time
time_diff = np.linspace(-5 * tau_stdp, 5 * tau_stdp, 50)

dW = Delta_W(pars, A_plus, A_minus, tau_stdp)

mySTDP_plot(A_plus, A_minus, tau_stdp, time_diff, dW)
