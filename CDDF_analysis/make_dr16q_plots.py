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

def do_cddf_occams(subdir: str = "CDDF_analysis", subdir_occams_upper: str = "CDDF_analysis", subdir_occams_lower: str = "CDDF_analysis"):
    """
    Make CDDF plots with systematics from Occams razors.
    """
    dla_data.noterdaeme_12_data()

    # (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]) in (6, N) shape
    cddf_all = np.loadtxt(os.path.join(subdir, "cddf_all.txt"))
    (_, N) = cddf_all.shape

    cddf68 = np.full((N, 2), fill_value=np.nan)
    cddf95 = np.full((N, 2), fill_value=np.nan)
    (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]) = cddf_all

    plot_cddf(l_N, cddf, cddf68, cddf95)

    # Occams lower and upper
    cddf_all = np.loadtxt(os.path.join(subdir_occams_upper, "cddf_all.txt"))
    (_, N) = cddf_all.shape

    cddf68 = np.full((N, 2), fill_value=np.nan)
    cddf95 = np.full((N, 2), fill_value=np.nan)
    (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]) = cddf_all

    plot_cddf(l_N, cddf, cddf68, cddf95, label="GP Occams upper", color="C1")

    # lower
    cddf_all = np.loadtxt(os.path.join(subdir_occams_lower, "cddf_all.txt"))
    (_, N) = cddf_all.shape

    cddf68 = np.full((N, 2), fill_value=np.nan)
    cddf95 = np.full((N, 2), fill_value=np.nan)
    (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]) = cddf_all

    plot_cddf(l_N, cddf, cddf68, cddf95, label="GP Occams lower", color="C2")
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    plt.tight_layout()
    save_figure(os.path.join(subdir, "cddf_gp_occams"))
    plt.clf()


def plot_cddf(l_N: np.ndarray, cddf: np.ndarray, cddf68: np.ndarray, cddf95: np.ndarray, lnhi_nbins: int = 30, lnhi_min: float = 20.0, lnhi_max: float = 23.0, color="blue", label="GP"):
    """
    Plot cddf from variables, using same methods as calc_cddf.plot_cddf
    """
    # Get the NHI bins
    l_nhi = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

    xerrs = (10 ** l_N - 10 ** l_nhi[:-1], 10 ** l_nhi[1:] - 10 ** l_N)

    # two sigma
    plt.fill_between(
        10 ** l_N, cddf95[:, 0], cddf95[:, 1], color="grey", alpha=0.5
    )

    # if has values
    yerr = (cddf - cddf68[:, 0], cddf68[:, 1] - cddf)
    ii = np.where(cddf68[:, 0] > 0.0)
    if np.size(ii) > 0:
        plt.errorbar(
            10 ** l_N[ii],
            cddf[ii],
            yerr=(yerr[0][ii], yerr[1][ii]),
            xerr=(xerrs[0][ii], xerrs[1][ii]),
            fmt="o",
            label=label,
            color=color,
        )

    # upper limits
    i2 = np.where(cddf68[:, 0] == 0)
    if np.size(i2) > 0:
        plt.errorbar(
            10 ** l_N[i2],
            cddf[i2] + yerr[1][i2],
            yerr=yerr[1][i2] / 2.0,
            xerr=(xerrs[0][i2], xerrs[1][i2]),
            fmt="o",
            label=None,
            uplims=True,
            color=color,
            lw=2,
        )
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
    plt.ylabel(r"$f(N_\mathrm{HI})$")

def do_dndx_occams(subdir: str = "CDDF_analysis", subdir_occams_upper: str = "CDDF_analysis", subdir_occams_lower: str = "CDDF_analysis"):
    """
    Make dNdX plots with systematics from Occams razors.
    """
    dla_data.dndx_not()
    dla_data.dndx_pro()

    # (z_cent, dNdX, dndx68[:,0],dndx68[:,1], dndx95[:,0],dndx95[:,1]) in (6, N) shape
    dndx_all = np.loadtxt(os.path.join(subdir, "dndx_all.txt"))
    (_, N) = dndx_all.shape

    dndx68 = np.full((N, 2), fill_value=np.nan)
    dndx95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, dNdX, dndx68[:,0], dndx68[:,1], dndx95[:,0], dndx95[:,1]) = dndx_all

    plot_line_density(z_cent, dNdX, dndx68, dndx95, color="C0")

    # Occams Upper
    dndx_all = np.loadtxt(os.path.join(subdir_occams_upper, "dndx_all.txt"))
    (_, N) = dndx_all.shape

    dndx68 = np.full((N, 2), fill_value=np.nan)
    dndx95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, dNdX, dndx68[:,0], dndx68[:,1], dndx95[:,0], dndx95[:,1]) = dndx_all

    plot_line_density(z_cent, dNdX, dndx68, dndx95, color="C1", label="GP Occams upper")

    # Occams Lower
    dndx_all = np.loadtxt(os.path.join(subdir_occams_lower, "dndx_all.txt"))
    (_, N) = dndx_all.shape

    dndx68 = np.full((N, 2), fill_value=np.nan)
    dndx95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, dNdX, dndx68[:,0], dndx68[:,1], dndx95[:,0], dndx95[:,1]) = dndx_all

    plot_line_density(z_cent, dNdX, dndx68, dndx95, color="C2", label="GP Occams lower")
    plt.legend(loc=0)
    plt.ylim(0,0.16)
    plt.tight_layout()
    save_figure(os.path.join(subdir,"dndx_occams"))
    plt.clf()


def plot_line_density(z_cent: np.ndarray, dNdX: np.ndarray, dndx68: np.ndarray, dndx95: np.ndarray, zmin: float = 2., zmax: float = 5., label: str = "GP", color="blue", bins_per_z: int = 6):
    """Plot the line density as a function of redshift"""
    # Get the redshifts
    nbins = np.max([int((zmax - zmin) * bins_per_z), 1])
    z_bins = np.linspace(zmin, zmax, nbins + 1)

    # hope this line works
    xerrs = (z_cent - z_bins[:-1], z_bins[1:] - z_cent)

    # 2 sigma contours.
    plt.fill_between(z_cent, dndx95[:, 0], dndx95[:, 1], color="grey", alpha=0.5)
    yerr = (dNdX - dndx68[:, 0], dndx68[:, 1] - dNdX)
    plt.errorbar(z_cent, dNdX, yerr=yerr, xerr=xerrs, fmt="o", color=color, label=label)
    plt.xlabel(r"z")
    plt.ylabel(r"dN/dX")
    plt.xlim(zmin, zmax)


def do_omega_dla_occams(subdir: str = "CDDF_analysis", subdir_occams_upper: str = "CDDF_analysis", subdir_occams_lower: str = "CDDF_analysis"):
    """
    Make OmegaDLA plots with systematics from Occams razors.
    """
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()

    # (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) in (6, N) shape
    omega_dla_all = np.loadtxt(os.path.join(subdir, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) = omega_dla_all

    plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C0")

    # Occams upper
    omega_dla_all = np.loadtxt(os.path.join(subdir_occams_upper, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) = omega_dla_all

    plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C1", label="GP Occams upper")

    # Occams lower
    omega_dla_all = np.loadtxt(os.path.join(subdir_occams_lower, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) = omega_dla_all

    plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C2", label="GP Occams lower")

    plt.legend(loc=0)
    plt.xlim(2,5)
    plt.ylim(0,2.5)
    plt.tight_layout()
    save_figure(os.path.join(subdir,"omega_occams"))
    plt.clf()

def do_omega_dla_XQ100(subdir: str = "CDDF_analysis"):
    """
    Make OmegaDLA plots with systematics from Occams razors.
    """
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()
    dla_data.xq100_omega()

    # (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) in (6, N) shape
    omega_dla_all = np.loadtxt(os.path.join(subdir, "omega_dla_all.txt"))
    (_, N) = omega_dla_all.shape

    omega_dla_68 = np.full((N, 2), fill_value=np.nan)
    omega_dla_95 = np.full((N, 2), fill_value=np.nan)
    (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]) = omega_dla_all

    plot_omega_dla(z_cent, omega_dla, omega_dla_68, omega_dla_95, color="C0")

    plt.legend(loc=0)
    plt.xlim(2,5)
    plt.ylim(0,2.5)
    plt.tight_layout()
    save_figure(os.path.join(subdir,"omega_xq100"))
    plt.clf()

def plot_omega_dla(z_cent: np.ndarray, omega_dla: np.ndarray, omega_dla_68: np.ndarray, omega_dla_95: np.ndarray, zmin: float = 2., zmax: float = 5., label: str = "GP", color: str = "blue", twosigma: bool = True, bins_per_z: int = 6):
    """Plot omega_DLA as a function of redshift, with full Bayesian errors"""
    nbins = np.max([int((zmax - zmin) * bins_per_z), 1])
    z_bins = np.linspace(zmin, zmax, nbins + 1)

    xerrs = (z_cent - z_bins[:-1], z_bins[1:] - z_cent)

    # import pdb
    # pdb.set_trace()

    if twosigma:
        plt.fill_between(
            z_cent,
            1000 * omega_dla_95[:, 0],
            1000 * omega_dla_95[:, 1],
            color="grey",
            alpha=0.5,
        )
    yerr = (
        omega_dla - omega_dla_68[:, 0],
        omega_dla_68[:, 1] - omega_dla,
    )
    plt.errorbar(z_cent, omega_dla, yerr=yerr, xerr=xerrs, fmt="s", label=label, color=color)
    plt.xlabel(r"z")
    plt.ylabel(r"$10^3 \times \Omega_\mathrm{DLA}$")
    plt.xlim(zmin, zmax)
