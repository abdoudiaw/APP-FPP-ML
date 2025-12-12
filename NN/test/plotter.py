import torch
import nn_learner
import collections
import numpy as np
import matplotlib.pyplot as plt
import collections
import matplotlib as mpl
import random
import sklearn.metrics
import numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np

def _metrics_safe(t, p):
    t = np.asarray(t, float).ravel()
    p = np.asarray(p, float).ravel()
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    mae  = np.mean(np.abs(p - t))
    # R² (safe if variance is ~0)
    denom = np.sum((t - t.mean())**2)
    r2 = 1.0 - (np.sum((p - t)**2) / (denom if denom > 0 else np.finfo(float).eps))
    rmse = np.sqrt(np.mean((p - t)**2))
    return r2, mae, rmse


def apply_font_scale(scale=2.0):
    import matplotlib as mpl
    S = lambda v: int(v * scale)
    mpl.rcParams['pdf.fonttype']  = 42           # editable text in PDF
    mpl.rcParams['ps.fonttype']   = 42
    mpl.rcParams['svg.fonttype']  = 'none'       # text stays text in SVG
    mpl.rcParams.update({
        'font.size':        S(10),  # base
        'axes.titlesize':   S(12),
        'axes.labelsize':   S(10),
        'xtick.labelsize':  S(9),
        'ytick.labelsize':  S(9),
        'legend.fontsize':  S(9),
        'figure.titlesize': S(12),
    })
    return S  # handy sizing helper if you need it locally



def plot_true_vs_pred_with_uncertainty(predictions, true, point_badness, err_info, labels, scale=1.0):
    K = predictions.shape[1]
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    S = apply_font_scale(scale)

    fig, axarr = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    axarr = axarr.flatten()

    for i, ax in enumerate(axarr[:K]):
        p = np.asarray(predictions[:, i])
        t = np.asarray(true[:, i])
        bad = np.asarray(point_badness[:, i])
        eb  = np.asarray(err_info[i])
        if eb.ndim == 0:
            eb = np.full_like(p, float(eb))

        mm = np.nanmin(np.concatenate([p, t]))
        xx = np.nanmax(np.concatenate([p, t]))
        margin = 0.05 * (xx - mm + 1e-12)
        xg = np.linspace(mm - margin, xx + margin, 200)

        colors = np.clip(bad, 0, 1)
        sc = ax.scatter(t, p, c='blue', s=S(9), cmap="plasma",alpha=0.8, linewidths=0, rasterized=True)
        eb_med = np.nanmedian(eb)
        
#
        ax.plot(xg, xg, c="0.3", lw=S(1))
        ax.fill_between(xg, xg-eb_med,   xg+eb_med,   alpha=0.18, color="0.6", label="±1σ")
        ax.fill_between(xg, xg-2*eb_med, xg+2*eb_med, alpha=0.10, color="0.6", label="±2σ")

        r2, mae, rmse = _metrics_safe(t, p)
        resid = p - t
        z = resid / np.maximum(eb, 1e-12)
        cov1 = np.mean(np.abs(z) <= 1.0)
        cov2 = np.mean(np.abs(z) <= 2.0)

#        ax.set_title(labels[i])
#        ax.set_xlabel("SOLPS data")
#        ax.set_ylabel("NN model")
        ax.set_xlim(mm - margin, xx + margin)
        ax.set_ylim(mm - margin, xx + margin)
        ax.grid(alpha=0.3)

    plt.show()

