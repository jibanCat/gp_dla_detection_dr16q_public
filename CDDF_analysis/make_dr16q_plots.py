"""
Make plots for the Multi-DLA paper
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from .qso_loader import QSOLoader, file_loader
from .set_parameters import *
from .dla_data import dla_data

from .effective_optical_depth import effective_optical_depth

cmap = cm.get_cmap("viridis")

# change fontsize
matplotlib.rcParams.update({"font.size": 14})

# matplotlib.use('PDF')

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)


def do_procedure_plots(qsos_full_int: QSOLoader, qsos_original: QSOLoader):
    # scaling factor between rest_wavelengths to pixels
    min_lambda = qsos_full_int.GP.min_lambda - 10
    max_lambda = qsos_full_int.GP.max_lambda + 10
    scale = 1 / (max_lambda - min_lambda)

    nv_wavelength = 1240.81
    oi_wavelength = 1305.53
    cii_wavelength = 1335.31
    siv_wavelength = 1399.8

    # compare different learned mus
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(
        qsos_full_int.GP.rest_wavelengths,
        qsos_full_int.GP.mu,
        label=r"re-trained $\mu$",
    )
    ax.plot(
        qsos_original.GP.rest_wavelengths,
        qsos_original.GP.mu,
        label=r"Ho-Bird-Garnett (2020) $\mu$",
        color="lightblue",
    )

    ax.legend()
    ax.set_xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $(\AA)$")
    ax.set_ylabel(r"normalised flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax02 = ax.twiny()
    ax02.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax02.set_xticklabels(
        [r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$", r"NV", r"OI", r"CII", r"SIV"]
    )
    plt.tight_layout()
    save_figure("mu_changes")
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    # compare different learned omegas
    ax.plot(
        qsos_full_int.GP.rest_wavelengths,
        np.exp(qsos_full_int.GP.log_omega),
        label=r"re-trained $\omega$",
    )
    ax.plot(
        qsos_original.GP.rest_wavelengths,
        np.exp(qsos_original.GP.log_omega),
        label=r"Ho-Bird-Garnett (2020) $\omega$",
        color="lightblue",
    )

    ax.legend()
    ax.set_xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
    ax.set_ylabel(r"normalised flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax12 = ax.twiny()
    ax12.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax12.set_xticklabels(
        [r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$", r"NV", r"OI", r"CII", r"SIV"]
    )

    plt.tight_layout()
    save_figure("omega_changes")
    plt.clf()

    # plotting covariance matrix
    min_lambda = qsos_full_int.GP.min_lambda
    max_lambda = qsos_full_int.GP.max_lambda
    scale = np.shape(qsos_full_int.GP.C)[0] / (max_lambda - min_lambda)

    lyg_wavelength = 972

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(qsos_full_int.GP.C, origin="lower")
    ax.set_xticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyg_wavelength - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ],
    )
    ax.set_xticklabels(
        [
            r"Ly $\infty$",
            r"Ly $\gamma$",
            r"Ly $\beta$",
            r"Ly $\alpha$",
            r"NV",
            r"OI",
            r"CII",
            r"SIV",
        ],
        rotation=45,
    )
    ax.set_yticks(
        [
            (lyman_limit - min_lambda) * scale,
            (lyg_wavelength - min_lambda) * scale,
            (lyb_wavelength - min_lambda) * scale,
            (lya_wavelength - min_lambda) * scale,
            (nv_wavelength - min_lambda) * scale,
            (oi_wavelength - min_lambda) * scale,
            (cii_wavelength - min_lambda) * scale,
            (siv_wavelength - min_lambda) * scale,
        ]
    )
    ax.set_yticklabels(
        [
            r"Ly $\infty$",
            r"Ly $\gamma$",
            r"Ly $\beta$",
            r"Ly $\alpha$",
            r"NV",
            r"OI",
            r"CII",
            r"SIV",
        ]
    )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_figure("covariance_matrix")
    plt.clf()
