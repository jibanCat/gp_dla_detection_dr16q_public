"""
QSOLoader for DR16Q GP catalogue
"""

from typing import Dict, List, Tuple

from collections import namedtuple, Counter

import numpy as np
from matplotlib import pyplot as plt
import h5py

from astropy.io import fits

from .set_parameters import *
from .qso_loader import QSOLoader, GPLoader


class QSOLoaderDR16Q(QSOLoader):
    def __init__(
        self,
        preloaded_file: str = "preloaded_qsos.mat",
        catalogue_file: str = "catalog.mat",
        learned_file: str = "learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        processed_file: str = "processed_qsos_dr12q.mat",
        dla_concordance: str = "dla_catalog",
        los_concordance: str = "los_catalog",
        snrs_file: str = "snrs_qsos_multi_meanflux_dr16q.mat",
        sub_dla: bool = True,
        sample_file: str = "dla_samples_a03_30000.mat",
        tau_sample_file: str = "tau_0_samples_30000.mat",
        occams_razor: float = 1 / 30,
        dist_file: str = "DR16Q_v4.fits",
        is_qso_final_cut: bool = True,
        class_person_cut: bool = True,
        z_source_cut: bool = True,
        zestimation_cut: bool = True,
    ):

        super().__init__(
            preloaded_file=preloaded_file,
            catalogue_file=catalogue_file,
            learned_file=learned_file,
            processed_file=processed_file,
            dla_concordance=dla_concordance,
            los_concordance=los_concordance,
            snrs_file=snrs_file,
            sub_dla=sub_dla,
            sample_file=sample_file,
            occams_razor=occams_razor,
        )

        # read the meanflux suppression samples
        self.tau_sample_file = h5py.File(tau_sample_file, "r")

        self.tau_0_samples = self.tau_sample_file["tau_0_samples"][:, 0]
        self.beta_samples = self.tau_sample_file["beta_samples"][:, 0]

        # read the DR16Q catalogue file
        self.dist_file = dist_file
        self.hdu = fits.open(dist_file)

        # filter: from DR16Q catalog -> our QSO samples
        # 1) first bring in the test_ind and nan_ind built in QSOLoader
        self.condition = np.ones(np.sum(self.test_ind), dtype=np.bool)
        # 2) filter out the disagreement in zestimation
        if zestimation_cut:
            self.zestimation_cut()
        # 3) filter out the rest of them with eBOSS cut
        self.eBOSS_cut(is_qso_final_cut, class_person_cut, z_source_cut)

        # reload the arrays with the eBOSS cut condition
        self._load_arrays()

    def _load_arrays(self):
        """
        Load GP processed file and catalog based on the self.condition
        """
        self.condition

        # test_set prior inds : organise arrays into the same order using selected test_inds
        # self.test_ind = self.processed_file['test_ind'][0, :].astype(np.bool) #size: (num_qsos, )
        # Assume you already have test_ind in the self
        assert "test_ind" in dir(self)
        self.test_real_index = np.nonzero(self.test_ind)[0]

        # load processed data
        self.model_posteriors = self.processed_file["model_posteriors"][()].T

        self.p_dlas = self.processed_file["p_dlas"][0, :]
        self.p_no_dlas = self.processed_file["p_no_dlas"][0, :]
        if self.sub_dla:
            self.p_no_dlas += self.model_posteriors[:, 1]
        self.log_priors_dla = self.processed_file["log_priors_dla"][0, :]
        self.min_z_dlas = self.processed_file["min_z_dlas"][0, :]
        self.max_z_dlas = self.processed_file["max_z_dlas"][0, :]
        if "MAP_log_nhis" in self.processed_file.keys():
            self.map_log_nhis = self.processed_file["MAP_log_nhis"][()].T
            self.map_z_dlas = self.processed_file["MAP_z_dlas"][()].T

        # [partial file] if we are reading a partial processed file trim down
        # the test_ind
        if self.p_dlas.shape[0] != self.test_real_index.shape[0]:
            print("[Warning] You have less samples in processed file than test_ind,")
            print(
                "          you are reading a partial file, otherwise, check your test_ind."
            )
            size = self.p_dlas.shape[0]
            self.test_ind[self.test_real_index[size:]] = False
            self.test_real_index = np.nonzero(self.test_ind)[0]
            assert self.test_real_index.shape[0] == self.p_dlas.shape[0]
            print("[Info] Reading first {} spectra.".format(size))

        # load snrs data
        try:
            self.snrs = self.snrs_file["snrs"][0, :]
        except TypeError as e:
            # this is to read the file computed by sbird's calc_cddf.compute_all_snrs function
            print(e)
            print(
                "[Warning] You are reading the snrs file from calc_cddf.compute_all_snrs."
            )
            self.snrs = self.snrs_file["snrs"][()]
        # [partial processed file but snrs not partial]
        if self.snrs.shape[0] != self.test_real_index.shape[0]:
            assert self.snrs.shape[0] == np.sum(
                self.processed_file["test_ind"][0, :].astype(np.bool)
            )
            print(
                "[Warning] Reading only first {} snrs.".format(
                    self.test_real_index.shape[0]
                )
            )
            self.snrs = self.snrs[: self.test_real_index.shape[0]]

        # store thing_ids based on test_set prior inds
        self.thing_ids = self.catalogue_file["thing_ids"][0, :].astype(np.int)
        self.thing_ids = self.thing_ids[self.test_ind]

        # plates, mjds, fiber_ids
        self.plates = self.catalogue_file["plates"][0, :].astype(np.int)
        self.mjds = self.catalogue_file["mjds"][0, :].astype(np.int)
        self.fiber_ids = self.catalogue_file["fiber_ids"][0, :].astype(np.int)

        self.plates = self.plates[self.test_ind]
        self.mjds = self.mjds[self.test_ind]
        self.fiber_ids = self.fiber_ids[self.test_ind]

        # store small arrays
        self.z_qsos = self.catalogue_file["z_qsos"][0, :]
        self.z_qsos = self.z_qsos[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        self.model_posteriors = self._occams_model_posteriors(
            self.model_posteriors, self.occams_razor
        )
        self.p_dlas = self.model_posteriors[:, 1 + self.sub_dla :].sum(axis=1)
        self.p_no_dlas = self.model_posteriors[:, : 1 + self.sub_dla].sum(axis=1)

        # build a MAP number of DLAs array
        # construct a reference array of model_posteriors in Roman's catalogue for computing ROC curve
        multi_p_dlas = self.model_posteriors  # shape : (num_qsos, 2 + num_dlas)

        dla_map_model_index = np.argmax(multi_p_dlas, axis=1)
        multi_p_dlas = multi_p_dlas[
            np.arange(multi_p_dlas.shape[0]), dla_map_model_index
        ]

        # get the number of DLAs with the highest val in model_posteriors
        dla_map_num_dla = dla_map_model_index
        if self.sub_dla:
            # (no dla, sub dla, DLA(1), DLA(2), ..., DLA(k))
            dla_map_num_dla = dla_map_num_dla - self.sub_dla
            dla_map_num_dla[dla_map_num_dla < 0] = 0

        self.multi_p_dlas = multi_p_dlas
        self.dla_map_model_index = dla_map_model_index
        self.dla_map_num_dla = dla_map_num_dla

        # construct an array based on model_posteriors, the array should be
        # [ {P(Mdla(m) | D)}j=1^m, P(Mdla(m+1) | D), ..., P(Mdla(k) | D) ]
        # This should be the reference p_dlas for multi-DLAs if we divide each los into num_dla pieces.
        model_posteriors_dla = np.copy(
            self.model_posteriors[:, 1 + self.sub_dla :].T
        )  # shape : (num_dlas, num_qsos)
        num_models = model_posteriors_dla.shape[0]
        num_qsos = model_posteriors_dla.shape[1]

        # build a mask array for assign P(Mdla(m) | D) until `m`th index
        indices = np.arange(1, num_models + 1, dtype=np.int8)[:, None] * np.ones(
            num_qsos, dtype=np.int8
        )
        multi_dla_map_num_dla = self.dla_map_num_dla * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )  # shape : (num_dlas, num_qsos) )
        multi_highest_posterior = self.multi_p_dlas * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )
        multi_p_no_dlas = self.p_no_dlas * np.ones(
            (num_models, num_qsos), dtype=np.int8
        )
        mask_inds = indices <= multi_dla_map_num_dla

        model_posteriors_dla[mask_inds] = multi_highest_posterior[mask_inds]
        self.model_posteriors_dla = model_posteriors_dla
        self.multi_p_no_dlas = multi_p_no_dlas

        # [filtering] Filter out the spectra based on the given condition
        self.filter_dla_spectra()

    def filter_dla_spectra(
        self,
    ):
        """
        Filter DLA based on the given condition.
        """
        # just re-write whole bunch of arrays here
        self.model_posteriors = self.model_posteriors[self.condition]
        self.p_dlas = self.p_dlas[self.condition]
        self.p_no_dlas = self.p_no_dlas[self.condition]
        self.log_priors_dla = self.log_priors_dla[self.condition]
        self.min_z_dlas = self.min_z_dlas[self.condition]
        self.max_z_dlas = self.max_z_dlas[self.condition]
        if "MAP_log_nhis" in self.processed_file.keys():
            self.map_z_dlas = self.map_z_dlas[self.condition]
            self.map_log_nhis = self.map_log_nhis[self.condition]

        # catalog file
        self.snrs = self.snrs[self.condition]
        self.thing_ids = self.thing_ids[self.condition]
        self.plates = self.plates[self.condition]
        self.mjds = self.mjds[self.condition]
        self.fiber_ids = self.fiber_ids[self.condition]
        self.z_qsos = self.z_qsos[self.condition]

        # some derived arrays
        self.multi_p_dlas = self.multi_p_dlas[self.condition]
        self.dla_map_model_index = self.dla_map_model_index[self.condition]
        self.dla_map_num_dla = self.dla_map_num_dla[self.condition]
        self.model_posteriors_dla = self.model_posteriors_dla.T[self.condition].T
        self.multi_p_no_dlas = self.multi_p_no_dlas.T[self.condition].T

    def eBOSS_cut(
        self,
        is_qso_final_cut: bool,
        class_person_cut: bool,
        z_source_cut: bool,
    ):
        """
        Apply some cuts defined in the eBOSS paper but did not implemented in build_catalog.mat
        """

        # the best redshift measurement column, only in DR16Q
        z_pca = self.hdu[1].data["Z_PCA"][self.test_ind]
        # make sure the distfile is the same as the one we used for catalog.mat
        assert np.all(np.abs(z_pca - self.z_qsos) < 1e-3)

        if is_qso_final_cut:
            is_qso_final = self.hdu[1].data["IS_QSO_FINAL"][self.test_ind]
            # IS_QSO_FINAL: -2 ~ 2
            # QSO: 1; Questionable QSO: 2
            # non QSO: -2 ~ 0
            # Details see 3.6 section of eBOSS DR16Q paper
            is_qso_condition = is_qso_final == 1

            print(
                "[Info] {} -> {} after setting IS_QSO_FINAL == 1.".format(
                    np.sum(self.condition), np.sum(is_qso_condition * self.condition)
                )
            )
            self.condition = self.condition * is_qso_condition

        if class_person_cut:
            class_person = self.hdu[1].data["CLASS_PERSON"][self.test_ind]
            # visual inspection classification of BAL
            # 0: no inspected; 1: star; 3: Quasar; 4: Galaxy
            # 30: BAL Quasar; 50: Blazar (?)
            # Details see Table 2 and Section 3.5 in DR16Q
            #
            # We aim to avoid BAL Quasars but keep those non-inspected
            human_qso_condition = ~((class_person != 3) & (class_person != 0))
            assert np.sum(human_qso_condition) == (
                np.sum(class_person == 3) + np.sum(class_person == 0)
            )

            print(
                "[Info] {} -> {} after setting class_person == 3 or 0.".format(
                    np.sum(self.condition), np.sum(human_qso_condition * self.condition)
                )
            )
            self.condition = self.condition * human_qso_condition

        if z_source_cut:
            source_z = self.hdu[1].data["SOURCE_Z"][self.test_ind]
            # Section 3.4: Z > 5 and source_z == "pipe" should be considered suspect
            z_source_condition = ~((source_z == "PIPE") & (z_pca > 5))

            print(
                "[Info] {} -> {} after filtering out SOURCE_Z: PIPE and Z > 5".format(
                    np.sum(self.condition), np.sum(z_source_condition * self.condition)
                )
            )
            self.condition = self.condition * z_source_condition

    def zestimation_cut(self, delta_z_qso: float = 0.2):
        """
        Cut spectra with zQSO disagreements
        """
        # the best redshift measurement column, only in DR16Q
        z_best = self.hdu[1].data["Z"][self.test_ind]

        z_pca = self.hdu[1].data["Z_PCA"][self.test_ind]
        z_pipe = self.hdu[1].data["Z_PIPE"][self.test_ind]
        z_vi = self.hdu[1].data["Z_VI"][self.test_ind]

        # make sure the distfile is the same as the one we used for catalog.mat
        assert np.all(np.abs(z_pca - self.z_qsos) < 1e-3)

        all_z_methods = [z_pca, z_pipe, z_vi]
        z_condition = np.ones_like(z_pca, dtype=np.bool)
        for z_method in all_z_methods:
            ind = np.abs(z_pca - z_method) < delta_z_qso

            ignore_ind = z_method == -1
            ind[ignore_ind] = True

            z_condition = ind * z_condition

        print(
            "[Info] {} -> {} after filtering out uncertain z measures.".format(
                np.sum(self.condition), np.sum(z_condition)
            )
        )
        self.z_condition = z_condition
        self.condition = self.condition * z_condition

    @staticmethod
    def prediction_parks2dict(dist_file: str, selected_thing_ids=None) -> Dict:
        """
        Get Parks' CNN predictions on DR16Q catalogue

        Split a single spectrum to several DLAs.

        No DLA for NaN in dla_confidence

        Z_PCA for zQSO.
        """
        hdu = fits.open(dist_file)

        # extract DLA information (exclude subDLA, lyb)
        ras = []
        decs = []
        plates = []
        mjds = []
        fiber_ids = []
        z_qsos = []
        dla_confidences = []
        z_dlas = []
        log_nhis = []
        thing_ids = []

        num_quasars = hdu[1].data["THING_ID"].shape[0]

        for i in range(num_quasars):
            # find number DLAs
            conf_dla = hdu[1].data["CONF_DLA"][i, :]
            num_dlas = np.sum(conf_dla != -1)

            assert num_dlas >= 0
            assert num_dlas <= 5

            # [thing_id] check if the thing_id is in the selected
            thing_id = hdu[1].data["THING_ID"][i]
            if thing_id not in selected_thing_ids:
                continue

            # has dla(s)
            if num_dlas > 0:
                for j in range(num_dlas):
                    # append basic quasar info
                    thing_ids.append(hdu[1].data["THING_ID"][i])
                    ras.append(hdu[1].data["RA"][i])
                    decs.append(hdu[1].data["DEC"][i])
                    plates.append(hdu[1].data["PLATE"][i])
                    mjds.append(hdu[1].data["MJD"][i])
                    fiber_ids.append(hdu[1].data["FIBERID"][i])
                    z_qsos.append(hdu[1].data["Z_PCA"][i])

                    # append the object (dla or lyb or subdla) info
                    dla_confidences.append(hdu[1].data["CONF_DLA"][i, j])
                    z_dlas.append(hdu[1].data["Z_DLA"][i, j])
                    log_nhis.append(hdu[1].data["NHI_DLA"][i, j])

                    assert dla_confidences[-1] >= 0
                    assert dla_confidences[-1] <= 1
                    assert log_nhis[-1] >= 20.3
                    assert z_qsos[-1] >= 0

            elif num_dlas == 0:
                # append basic quasar info
                thing_ids.append(hdu[1].data["THING_ID"][i])
                ras.append(hdu[1].data["RA"][i])
                decs.append(hdu[1].data["DEC"][i])
                plates.append(hdu[1].data["PLATE"][i])
                mjds.append(hdu[1].data["MJD"][i])
                fiber_ids.append(hdu[1].data["FIBERID"][i])
                z_qsos.append(hdu[1].data["Z_PCA"][i])

                # append the object (dla or lyb or subdla) info
                dla_confidences.append(np.nan)
                z_dlas.append(np.nan)
                log_nhis.append(np.nan)

        dict_parks = {
            "thing_ids": np.array(thing_ids).astype(np.int),
            "ras": np.array(ras),
            "decs": np.array(decs),
            "plates": np.array(plates).astype(np.int),
            "mjds": np.array(mjds).astype(np.int),
            "fiber_ids": np.array(fiber_ids).astype(np.int),
            "z_qso": np.array(z_qsos),
            "dla_confidences": np.array(dla_confidences),
            "z_dlas": np.array(z_dlas),
            "log_nhis": np.array(log_nhis),
        }

        return dict_parks

    def _get_parks_estimations(
        self,
        dla_parks: str = "DR16Q_v4.fits",
        p_thresh: float = 0.97,
        lyb: bool = False,
        prior: bool = False,
        search_range_from_ours: bool = False,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Get z_dlas and log_nhis from Parks' (2018) estimations
        """
        # make sure we are reading the DR16Q file
        assert dla_parks in self.dist_file

        if "dict_parks" not in dir(self):
            self.dict_parks = self.prediction_parks2dict(
                self.dist_file, selected_thing_ids=self.thing_ids
            )

        if "p_thresh" in self.dict_parks.keys():
            if self.dict_parks["p_thresh"] == p_thresh:
                thing_ids = self.dict_parks["cddf_thing_ids"]
                log_nhis = self.dict_parks["cddf_log_nhis"]
                z_dlas = self.dict_parks["cddf_z_dlas"]
                min_z_dlas = self.dict_parks["min_z_dlas"]
                max_z_dlas = self.dict_parks["max_z_dlas"]
                snrs = self.dict_parks["snrs"]
                all_snrs = self.dict_parks["all_snrs"]
                p_dlas = self.dict_parks["cddf_p_dlas"]
                p_thresh = self.dict_parks["p_thresh"]

                return (
                    thing_ids,
                    log_nhis,
                    z_dlas,
                    min_z_dlas,
                    max_z_dlas,
                    snrs,
                    all_snrs,
                    p_dlas,
                )

        dict_parks = self.dict_parks

        # use thing_ids as an identifier
        # use zPCA as zQSOs, as said in DR16Q paper
        thing_ids = dict_parks["thing_ids"]
        z_qsos = dict_parks["z_qso"]

        # fixed range of the sightline ranging from 911A-1215A in rest-frame
        # we should include all sightlines in the dataset

        assert np.all(np.isin(thing_ids, self.thing_ids))

        # for loop to get snrs from sbird's snrs file
        all_snrs = np.zeros(thing_ids.shape)

        for i, thing_id in enumerate(thing_ids):
            real_index = np.where(self.thing_ids == thing_id)[0][0]
            all_snrs[i] = self.snrs[real_index]

        # the searching range for the DLAs.
        if lyb:
            print("Searching min_z_dla: lyb ... ")
            min_z_dlas = (1 + z_qsos) * lyb_wavelength / lya_wavelength - 1
        else:
            min_z_dlas = (1 + z_qsos) * lyman_limit / lya_wavelength - 1
        print("Searching max_z_dla: lya ... ")
        max_z_dlas = (1 + z_qsos) * lya_wavelength / lya_wavelength - 1

        # get search range from the processed file
        # redo the full searching range
        if search_range_from_ours:
            print("Using our own searching range")
            min_z_dlas = np.zeros(thing_ids.shape)
            max_z_dlas = np.zeros(thing_ids.shape)

            for i, thing_id in enumerate(thing_ids):
                real_index = np.where(self.thing_ids == thing_id)[0][0]

                min_z_dlas[i] = self.min_z_dlas[real_index]
                max_z_dlas[i] = self.max_z_dlas[real_index]

        # get DLA properties
        # note: the following indices are DLA-only
        dla_inds = (
            dict_parks["dla_confidences"] > 0.005
        )  # use p_thresh=0.005 to filter out non-DLA spectra and
        # speed up the computation

        thing_ids = thing_ids[dla_inds]
        log_nhis = dict_parks["log_nhis"][dla_inds]
        z_dlas = dict_parks["z_dlas"][dla_inds]
        z_qsos = dict_parks["z_qso"][dla_inds]
        p_dlas = dict_parks["dla_confidences"][dla_inds]

        # for loop to get snrs from sbird's snrs file
        # Note arrays here are for DLA only
        snrs = np.zeros(thing_ids.shape)
        log_priors_dla = np.zeros(thing_ids.shape)

        for i, thing_id in enumerate(thing_ids):
            real_index = np.where(self.thing_ids == thing_id)[0][0]

            snrs[i] = self.snrs[real_index]
            log_priors_dla[i] = self.log_priors_dla[real_index]

        # re-calculate dla_confidence based on prior of DLAs given z_qsos
        if prior:
            print("Applying our DLA prior ...")
            p_dlas = p_dlas * np.exp(log_priors_dla)
            p_dlas = p_dlas / np.max(p_dlas)

        dla_inds = p_dlas > p_thresh

        thing_ids = thing_ids[dla_inds]
        log_nhis = log_nhis[dla_inds]
        z_dlas = z_dlas[dla_inds]
        z_qsos = z_qsos[dla_inds]
        p_dlas = p_dlas[dla_inds]
        snrs = snrs[dla_inds]
        log_priors_dla = log_priors_dla[dla_inds]

        # get rid of z_dlas larger than z_qsos or lower than lyman limit
        if lyb:
            z_cut_inds = z_dlas > ((1 + z_qsos) * lyb_wavelength / lya_wavelength - 1)
        else:
            z_cut_inds = z_dlas > ((1 + z_qsos) * lyman_limit / lya_wavelength - 1)

        z_cut_inds = np.logical_and(
            z_cut_inds, (z_dlas < ((1 + z_qsos) * lya_wavelength / lya_wavelength - 1))
        )

        if search_range_from_ours:
            # we are not sure if the search range is lyb in the processed file
            assert lyb is False

            # for loop to get min z_dlas and max z_dlas search range from processed data
            _min_z_dlas = np.zeros(thing_ids.shape)
            _max_z_dlas = np.zeros(thing_ids.shape)

            for i, thing_id in enumerate(thing_ids):
                real_index = np.where(self.thing_ids == thing_id)[0][0]

                _min_z_dlas[i] = self.min_z_dlas[real_index]
                _max_z_dlas[i] = self.max_z_dlas[real_index]

            # # Parks chap 3.2: fixed range of the sightline ranging from 900A-1346A in rest-frame
            # min_z_dlas = (1 + z_qsos) *  900   / lya_wavelength - 1
            # max_z_dlas = (1 + z_qsos) *  1346  / lya_wavelength - 1

            z_cut_inds = z_dlas >= _min_z_dlas
            z_cut_inds = np.logical_and(z_cut_inds, (z_dlas <= _max_z_dlas))

        thing_ids = thing_ids[z_cut_inds]
        log_nhis = log_nhis[z_cut_inds]
        z_dlas = z_dlas[z_cut_inds]
        z_qsos = z_qsos[z_cut_inds]
        p_dlas = p_dlas[z_cut_inds]
        snrs = snrs[z_cut_inds]
        log_priors_dla = log_priors_dla[z_cut_inds]

        self.dict_parks["cddf_thing_ids"] = thing_ids
        self.dict_parks["cddf_log_nhis"] = log_nhis
        self.dict_parks["cddf_z_dlas"] = z_dlas
        self.dict_parks["min_z_dlas"] = min_z_dlas
        self.dict_parks["max_z_dlas"] = max_z_dlas
        self.dict_parks["snrs"] = snrs
        self.dict_parks["all_snrs"] = all_snrs
        self.dict_parks["cddf_p_dlas"] = p_dlas
        self.dict_parks["p_thresh"] = p_thresh

        return (
            thing_ids,
            log_nhis,
            z_dlas,
            min_z_dlas,
            max_z_dlas,
            snrs,
            all_snrs,
            p_dlas,
        )

    def plot_this_mu(
        self,
        nspec,
        suppressed=True,
        num_voigt_lines=3,
        num_forest_lines=31,
        Parks=False,
        dla_parks=None,
        label="",
        new_fig=True,
        color="red",
        conditional_gp: bool = False,
        color_cond: str = "lightblue",
    ):
        """
        Plot the spectrum with the dla model

        Parameters:
        ----
        nspec (int) : index of the spectrum in the catalogue
        suppressed (bool) : apply Lyman series suppression to the mean-flux or not
        num_voigt_lines (int, min=1, max=31) : how many members of Lyman series in the DLA Voigt profile
        number_forest_lines (int) : how many members of Lymans series considered in the froest
        Parks (bool) : whether to plot Parks' results
        dla_parks (str) : if Parks=True, specify the path to Parks' `prediction_DR12.json`

        Returns:
        ----
        rest_wavelengths : rest wavelengths for the DLA model
        this_mu : the DLA model
        map_z_dlas : MAP z_dla values
        map_log_nhis : MAP log NHI values
        """
        NotImplementedError

    def load_dla_parks(
        self,
        dla_parks: str = "DR16Q_v4.fits",
        p_thresh: float = 0.98,
        release: str = "dr12",
        multi_dla: bool = True,
        num_dla: int = 2,
    ):
        """
        load Parks CNN predictions out of DR16Q catalogue

        Note: we have to consider DLAs from the same sightlines as different objects

        Parameters:
        ----
        dla_parks (str) : the filename of DR16Q catalogue
        p_thresh (float): the minimum probability to be considered as a DLA in Parks CNN
        release (str)
        multi_dla (bool): whether or not we want to construct multi-dla index
        num_dla (int)   : number of dla we want to consider if we are considering multi-dlas

        Note:
        ---
        We only load those spectra in our GP catalogue
        """
        dict_parks = self.prediction_parks2dict(
            dla_parks, selected_thing_ids=self.thing_ids
        )

        # construct an array of unique ids for los
        # Note: keep spectral ID here to avoid thing_id to be -1
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids = self.make_unique_id(
            dict_parks["plates"], dict_parks["mjds"], dict_parks["fiber_ids"]
        )
        assert unique_ids.dtype is np.dtype("int64")
        assert self.unique_ids.dtype is np.dtype("int64")

        # TODO: make the naming of variables more consistent
        raw_unique_ids = unique_ids
        raw_thing_ids = dict_parks["thing_ids"]
        raw_z_dlas = dict_parks["z_dlas"]
        raw_log_nhis = dict_parks["log_nhis"]
        raw_dla_confidences = dict_parks["dla_confidences"]

        real_index_los = np.where(np.in1d(self.unique_ids, unique_ids))[0]

        unique_ids_los = self.unique_ids[real_index_los]
        thing_ids_los = self.thing_ids[real_index_los]
        assert (
            np.unique(unique_ids_los).shape[0] == unique_ids_los.shape[0]
        )  # make sure we don't double count los

        # construct an array of unique ids for dlas
        dla_inds = dict_parks["dla_confidences"] > p_thresh

        real_index_dla = np.where(np.in1d(self.unique_ids, unique_ids[dla_inds]))[
            0
        ]  # Note that in this step we lose
        # the info about multi-DLA since
        # we are counting based on los

        unique_ids_dla = self.unique_ids[real_index_dla]
        thing_ids_dla = self.thing_ids[real_index_dla]

        # Construct a list of sub-los index and dla detection based on sub-los.
        # This is a relatively complicate loop and it's hard to understand philosophically.
        # It's better to write an explaination in the paper.
        if multi_dla:
            self.multi_unique_ids = self.make_multi_unique_id(
                num_dla, self.plates, self.mjds, self.fiber_ids
            )
            multi_unique_ids = self.make_multi_unique_id(
                num_dla,
                dict_parks["plates"],
                dict_parks["mjds"],
                dict_parks["fiber_ids"],
            )  # note here some index repeated
            # more than num_dla times

            multi_real_index_los = np.where(
                np.in1d(self.multi_unique_ids, multi_unique_ids)
            )[
                0
            ]  # here we have a real_index array
            # exactly repeat num_dla times

            multi_unique_ids_los = self.multi_unique_ids[multi_real_index_los]

            self.multi_thing_ids = self.make_array_multi(num_dla, self.thing_ids)
            multi_thing_ids_los = self.multi_thing_ids[multi_real_index_los]

            # loop over unique_ids to assign DLA detection to sub-los
            # Note: here we ignore the z_dla of DLAs.
            dla_multi_inds = np.zeros(multi_unique_ids_los.shape, dtype=bool)
            for uid in np.unique(multi_unique_ids_los):
                k_dlas = (
                    dict_parks["dla_confidences"][unique_ids == uid] > p_thresh
                ).sum()

                k_dlas_val = np.zeros(num_dla, dtype=bool)
                k_dlas_val[:k_dlas] = True  # assigning True until DLA(k)

                # assign DLA detections to the unique_ids of sub-los
                dla_multi_inds[multi_unique_ids_los == uid] = k_dlas_val
                assert (
                    multi_unique_ids_los[multi_unique_ids_los == uid].shape[0]
                    == num_dla
                )

            multi_real_index_dla = multi_real_index_los[dla_multi_inds]
            multi_unique_ids_dla = multi_unique_ids_los[dla_multi_inds]
            multi_thing_ids_dla = multi_thing_ids_los[dla_multi_inds]

            # store data in named tuple under self
            dla_catalog = namedtuple(
                "dla_catalog_parks",
                [
                    "real_index",
                    "real_index_los",
                    "thing_ids",
                    "thing_ids_los",
                    "unique_ids",
                    "unique_ids_los",
                    "multi_real_index_dla",
                    "multi_real_index_los",
                    "multi_thing_ids_dla",
                    "multi_thing_ids_los",
                    "multi_unique_ids_dla",
                    "multi_unique_ids_los",
                    "release",
                    "num_dla",
                    "raw_unique_ids",
                    "raw_z_dlas",
                    "raw_log_nhis",
                    "raw_dla_confidences",
                ],
            )
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla,
                real_index_los=real_index_los,
                thing_ids=thing_ids_dla,
                thing_ids_los=thing_ids_los,
                unique_ids=unique_ids_dla,
                unique_ids_los=unique_ids_los,
                multi_real_index_dla=multi_real_index_dla,
                multi_real_index_los=multi_real_index_los,
                multi_thing_ids_dla=multi_thing_ids_dla,
                multi_thing_ids_los=multi_thing_ids_los,
                multi_unique_ids_dla=multi_unique_ids_dla,
                multi_unique_ids_los=multi_unique_ids_los,
                release=release,
                num_dla=num_dla,
                raw_unique_ids=raw_unique_ids,
                raw_z_dlas=raw_z_dlas,
                raw_log_nhis=raw_log_nhis,
                raw_dla_confidences=raw_dla_confidences,
            )

        else:
            dla_catalog = namedtuple(
                "dla_catalog_parks",
                [
                    "real_index",
                    "real_index_los",
                    "thing_ids",
                    "thing_ids_los",
                    "unique_ids",
                    "unique_ids_los",
                    "release",
                    "raw_unique_ids",
                    "raw_z_dlas",
                    "raw_log_nhis",
                    "raw_dla_confidences",
                ],
            )
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla,
                real_index_los=real_index_los,
                thing_ids=thing_ids_dla,
                thing_ids_los=thing_ids_los,
                unique_ids=unique_ids_dla,
                unique_ids_los=unique_ids_los,
                release=release,
                raw_unique_ids=raw_unique_ids,
                raw_z_dlas=raw_z_dlas,
                raw_log_nhis=raw_log_nhis,
                raw_dla_confidences=raw_dla_confidences,
            )
