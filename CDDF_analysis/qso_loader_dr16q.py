"""
QSOLoader for DR16Q GP catalogue
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py

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
        self.test_real_index = np.nonzero( self.test_ind )[0]        

        # load processed data
        self.model_posteriors = self.processed_file['model_posteriors'][()].T

        self.p_dlas           = self.processed_file['p_dlas'][0, :]
        self.p_no_dlas        = self.processed_file['p_no_dlas'][0, :]
        if self.sub_dla:
            self.p_no_dlas += self.model_posteriors[:, 1]
        self.log_priors_dla   = self.processed_file['log_priors_dla'][0, :]
        self.min_z_dlas       = self.processed_file['min_z_dlas'][0, :]
        self.max_z_dlas       = self.processed_file['max_z_dlas'][0, :]
        if 'MAP_log_nhis' in self.processed_file.keys():
            self.map_log_nhis = self.processed_file['MAP_log_nhis'][()].T
            self.map_z_dlas   = self.processed_file['MAP_z_dlas'][()].T


        # [partial file] if we are reading a partial processed file trim down
        # the test_ind
        if self.p_dlas.shape[0] != self.test_real_index.shape[0]:
            print("[Warning] You have less samples in processed file than test_ind,")
            print("          you are reading a partial file, otherwise, check your test_ind.")
            size = self.p_dlas.shape[0]
            self.test_ind[self.test_real_index[size:]] = False
            self.test_real_index = np.nonzero(self.test_ind)[0]
            assert self.test_real_index.shape[0] == self.p_dlas.shape[0]
            print("[Info] Reading first {} spectra.".format(size))

        # load snrs data
        try:
            self.snrs         = self.snrs_file['snrs'][0, :]
        except TypeError as e:
            # this is to read the file computed by sbird's calc_cddf.compute_all_snrs function
            print(e)
            print("[Warning] You are reading the snrs file from calc_cddf.compute_all_snrs.")
            self.snrs         = self.snrs_file['snrs'][()] 
        # [partial processed file but snrs not partial]
        if self.snrs.shape[0] != self.test_real_index.shape[0]:
            assert self.snrs.shape[0] == np.sum(self.processed_file['test_ind'][0, :].astype(np.bool))
            print("[Warning] Reading only first {} snrs.".format(self.test_real_index.shape[0]))
            self.snrs = self.snrs[:self.test_real_index.shape[0]]

        # store thing_ids based on test_set prior inds
        self.thing_ids = self.catalogue_file['thing_ids'][0, :].astype(np.int)
        self.thing_ids = self.thing_ids[self.test_ind]

        # plates, mjds, fiber_ids
        self.plates    = self.catalogue_file['plates'][0, :].astype(np.int)
        self.mjds      = self.catalogue_file['mjds'][0, :].astype(np.int)
        self.fiber_ids = self.catalogue_file['fiber_ids'][0, :].astype(np.int)

        self.plates    = self.plates[self.test_ind]
        self.mjds      = self.mjds[self.test_ind]
        self.fiber_ids = self.fiber_ids[self.test_ind]

        # store small arrays
        self.z_qsos     = self.catalogue_file['z_qsos'][0, :]
        self.z_qsos = self.z_qsos[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        self.model_posteriors = self._occams_model_posteriors(self.model_posteriors, self.occams_razor)
        self.p_dlas    = self.model_posteriors[:, 1+self.sub_dla:].sum(axis=1)
        self.p_no_dlas = self.model_posteriors[:, :1+self.sub_dla].sum(axis=1)

        # build a MAP number of DLAs array
        # construct a reference array of model_posteriors in Roman's catalogue for computing ROC curve
        multi_p_dlas    = self.model_posteriors # shape : (num_qsos, 2 + num_dlas)

        dla_map_model_index = np.argmax( multi_p_dlas, axis=1 )
        multi_p_dlas = multi_p_dlas[ np.arange(multi_p_dlas.shape[0]), dla_map_model_index ]

        # get the number of DLAs with the highest val in model_posteriors
        dla_map_num_dla = dla_map_model_index
        if self.sub_dla:
            # (no dla, sub dla, DLA(1), DLA(2), ..., DLA(k))
            dla_map_num_dla = dla_map_num_dla - self.sub_dla
            dla_map_num_dla[dla_map_num_dla < 0] = 0

        self.multi_p_dlas        = multi_p_dlas
        self.dla_map_model_index = dla_map_model_index
        self.dla_map_num_dla     = dla_map_num_dla

        # construct an array based on model_posteriors, the array should be
        # [ {P(Mdla(m) | D)}j=1^m, P(Mdla(m+1) | D), ..., P(Mdla(k) | D) ]
        # This should be the reference p_dlas for multi-DLAs if we divide each los into num_dla pieces.
        model_posteriors_dla = np.copy( self.model_posteriors[:, 1 + self.sub_dla:].T ) # shape : (num_dlas, num_qsos)
        num_models = model_posteriors_dla.shape[0]
        num_qsos   = model_posteriors_dla.shape[1]

        # build a mask array for assign P(Mdla(m) | D) until `m`th index
        indices = (np.arange(1, num_models + 1, dtype=np.int8)[:, None] * np.ones( num_qsos, dtype=np.int8 ))
        multi_dla_map_num_dla   = (self.dla_map_num_dla * np.ones( (num_models, num_qsos), dtype=np.int8 )) # shape : (num_dlas, num_qsos) )
        multi_highest_posterior = (self.multi_p_dlas * np.ones( (num_models, num_qsos), dtype=np.int8 ))
        multi_p_no_dlas         = (self.p_no_dlas    * np.ones( (num_models, num_qsos), dtype=np.int8 ))
        mask_inds = indices <= multi_dla_map_num_dla
        
        model_posteriors_dla[mask_inds] = multi_highest_posterior[mask_inds]
        self.model_posteriors_dla = model_posteriors_dla
        self.multi_p_no_dlas      = multi_p_no_dlas

        # [filtering] Filter out the spectra based on the given condition
        self.filter_dla_spectra()
    
    def filter_dla_spectra(self, ):
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
        if 'MAP_log_nhis' in self.processed_file.keys():
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
        self, is_qso_final_cut: bool, class_person_cut: bool, z_source_cut: bool,
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


    def load_dla_parks(self, p_thresh: float = 0.97):
        NotImplementedError

    def _get_parks_estimations(self, p_thresh: float = 0.97):
        '''
        Get z_dlas and log_nhis from Parks' (2018) estimations
        '''
        if 'dict_parks' not in dir(self):
            self.dict_parks = self.prediction_json2dict(dla_parks)

        if 'p_thresh' in self.dict_parks.keys():
            if self.dict_parks['p_thresh'] == p_thresh:
                unique_ids  = self.dict_parks['unique_ids']
                log_nhis    = self.dict_parks['cddf_log_nhis']  
                z_dlas      = self.dict_parks['cddf_z_dlas']    
                min_z_dlas  = self.dict_parks['min_z_dlas']
                max_z_dlas  = self.dict_parks['max_z_dlas']
                snrs        = self.dict_parks['snrs']      
                all_snrs    = self.dict_parks['all_snrs']  
                p_dlas      = self.dict_parks['cddf_p_dlas']    
                p_thresh    = self.dict_parks['p_thresh']  

                return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas

        dict_parks = self.dict_parks

        # construct an array of unique ids for los
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'] ) 
        assert unique_ids.dtype is np.dtype('int64')
        assert self.unique_ids.dtype is np.dtype('int64')

        # fixed range of the sightline ranging from 911A-1215A in rest-frame
        # we should include all sightlines in the dataset        
        roman_inds = np.isin(unique_ids, self.unique_ids)

        z_qsos     = dict_parks['z_qso'][roman_inds]
        uids       = unique_ids[roman_inds]
        
        uids, indices = np.unique( uids, return_index=True )

        # for loop to get snrs from sbird's snrs file
        all_snrs           = np.zeros( uids.shape )

        for i,uid in enumerate(uids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            all_snrs[i]           = self.snrs[real_index]

        z_qsos     = z_qsos[indices]

        min_z_dlas = (1 + z_qsos) *  lyman_limit  / lya_wavelength - 1
        max_z_dlas = (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1

        # get DLA properties
        # note: the following indices are DLA-only
        dla_inds = dict_parks['dla_confidences'] > 0.005 # use p_thresh=0.005 to filter out non-DLA spectra and 
                                                         # speed up the computation

        unique_ids = unique_ids[dla_inds]
        log_nhis   = dict_parks['log_nhis'][dla_inds]
        z_dlas     = dict_parks['z_dlas'][dla_inds]
        z_qsos     = dict_parks['z_qso'][dla_inds]
        p_dlas     = dict_parks['dla_confidences'][dla_inds]

        # check if all ids are in Roman's sample
        roman_inds = np.isin(unique_ids, self.unique_ids)
        unique_ids = unique_ids[roman_inds]
        log_nhis   = log_nhis[roman_inds]
        z_dlas     = z_dlas[roman_inds]
        z_qsos     = z_qsos[roman_inds]
        p_dlas     = p_dlas[roman_inds]

        # for loop to get snrs from sbird's snrs file
        snrs           = np.zeros( unique_ids.shape )
        log_priors_dla = np.zeros( unique_ids.shape )

        for i,uid in enumerate(unique_ids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            snrs[i]           = self.snrs[real_index]
            log_priors_dla[i] = self.log_priors_dla[real_index]

        # re-calculate dla_confidence based on prior of DLAs given z_qsos
        if prior:
            p_dlas = p_dlas * np.exp(log_priors_dla)
            p_dlas = p_dlas / np.max(p_dlas)

        dla_inds = p_dlas > p_thresh

        unique_ids     = unique_ids[dla_inds]
        log_nhis       = log_nhis[dla_inds]
        z_dlas         = z_dlas[dla_inds]
        z_qsos         = z_qsos[dla_inds]
        p_dlas         = p_dlas[dla_inds]
        snrs           = snrs[dla_inds]
        log_priors_dla = log_priors_dla[dla_inds]

        # get rid of z_dlas larger than z_qsos or lower than lyman limit
        z_cut_inds = (
            z_dlas > ((1 + z_qsos) *  lyman_limit  / lya_wavelength - 1) ) 
        z_cut_inds = np.logical_and(
            z_cut_inds, (z_dlas < ( (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1 )) )

        unique_ids     = unique_ids[z_cut_inds]
        log_nhis       = log_nhis[z_cut_inds]
        z_dlas         = z_dlas[z_cut_inds]
        z_qsos         = z_qsos[z_cut_inds]
        p_dlas         = p_dlas[z_cut_inds]
        snrs           = snrs[z_cut_inds]
        log_priors_dla = log_priors_dla[z_cut_inds]

        # # for loop to get min z_dlas and max z_dlas search range from processed data
        # min_z_dlas = np.zeros( unique_ids.shape )
        # max_z_dlas = np.zeros( unique_ids.shape )

        # for i,uid in enumerate(unique_ids):
        #     real_index = np.where( self.unique_ids == uid )[0][0]

        #     min_z_dlas[i] = self.min_z_dlas[real_index]
        #     max_z_dlas[i] = self.max_z_dlas[real_index]

        # # Parks chap 3.2: fixed range of the sightline ranging from 900A-1346A in rest-frame
        # min_z_dlas = (1 + z_qsos) *  900   / lya_wavelength - 1
        # max_z_dlas = (1 + z_qsos) *  1346  / lya_wavelength - 1

        # assert np.all( ( z_dlas < max_z_dlas[0] ) & (z_dlas > min_z_dlas[0]) )

        self.dict_parks['unique_ids']    = unique_ids
        self.dict_parks['cddf_log_nhis'] = log_nhis
        self.dict_parks['cddf_z_dlas']   = z_dlas
        self.dict_parks['min_z_dlas']    = min_z_dlas 
        self.dict_parks['max_z_dlas']    = max_z_dlas
        self.dict_parks['snrs']          = snrs
        self.dict_parks['all_snrs']      = all_snrs
        self.dict_parks['cddf_p_dlas']   = p_dlas
        self.dict_parks['p_thresh']      = p_thresh

        return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas


    def plot_this_mu(self, nspec, suppressed=True, num_voigt_lines=3, num_forest_lines=31, Parks=False, dla_parks=None, 
        label="", new_fig=True, color="red", conditional_gp: bool = False, color_cond: str = "lightblue"):
        '''
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
        '''
        NotImplementedError