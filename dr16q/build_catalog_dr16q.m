% build_catalogs_dr16q: loads existing QSO and DLA catalogs, applies some
% initial filters, and creates a list of spectra to download from SDSS
%
% ZWARNING: ensure we exclude those spectra with bad redshift status reported

% spec ID in integer : plates * 10**9 + mjds * 10**4 + fiber_ids
% Note: thingIDs between DR16Q and DR12Q are unmatched, but specID are matched. 
convert_unique_id = @(plate, mjd, fiber_id) ...
  (uint64(plate) * 10^9 + uint64(mjd) * 10^4 + uint64(fiber_id));

% load QSO catalogs
release = 'dr12q';
dr12_catalog = ...
    fitsread(sprintf('%s/DR12Q.fits', distfiles_directory(release)), ...
             'binarytable');

release = 'dr16q';
dr16_catalog = ...
  fitsread(sprintf('%s/DR16Q_v4.fits', distfiles_directory(release)), ...
              'binarytable');

% extract basic QSO information from DR16Q catalog
sdss_names       =  dr16_catalog{1};
ras              =  dr16_catalog{2};
decs             =  dr16_catalog{3};
thing_ids        =  dr16_catalog{16};
plates           =  dr16_catalog{4};
mjds             =  dr16_catalog{5};
fiber_ids        =  dr16_catalog{6};
z_qsos           =  dr16_catalog{27};         % Best available redshift taken from
                                              % Z VI, Z PIPE, Z DR12Q, Z DR7Q SCH, or Z DR6Q HW
zwarning         =  dr16_catalog{30};
bal_visual_flags = (dr16_catalog{57} > 0.75); % 99,856 spectra with BAL PROB ≥ 0.75
is_qso_dr12q     =  dr16_catalog{21};         % Flag indicating if an object was a quasar in DR12Q

num_quasars = numel(z_qsos);

% extract basic QSO information from DR12Q catalog
plates_dr12q           =  dr12_catalog{5};
mjds_dr12q             =  dr12_catalog{6};
fiber_ids_dr12q        =  dr12_catalog{7};
z_qsos_dr12q           =  dr12_catalog{8};

% [match DR12Q and DR16Q] make unique IDs
unique_ids       = convert_unique_id(plates,       mjds,       fiber_ids);
unique_ids_dr12q = convert_unique_id(plates_dr12q, mjds_dr12q, fiber_ids_dr12q);

% determine which objects in DR16Q are in DR12Q, using spec IDs
in_dr12 = ismember(unique_ids,       unique_ids_dr12q);
in_dr16 = ismember(unique_ids_dr12q, unique_ids);
% make sure matching specIDs gives the same result as using eBOSS's flag
assert(sum(in_dr12) == sum(is_qso_dr12q == 1))

[num_dr16q, ~] = size(in_dr12);

assert(num_dr16q == num_quasars)
assert(sum(in_dr12) == sum(in_dr16))

% [match DR12Q and DR16Q] to be safe, filter out unmatched zQSOs.
% The DLA flags could be different if the zQSOs are different.
delta_z_qsos = abs(z_qsos(in_dr12) - z_qsos_dr12q(in_dr16));
z_qso_diff_in_dr12 = (delta_z_qsos < 1e-2); % saved for learning script
% create another in_dr12 to avoid zQSO difference
in_dr12_z = false(num_quasars, 1);
in_dr12_z(in_dr12) = z_qso_diff_in_dr12;

assert(sum(in_dr12_z) == sum(z_qso_diff_in_dr12))

% to track reasons for filtering out QSOs
filter_flags = zeros(num_quasars, 1, 'uint8');

% filtering bit 0: z_QSO < 2.15
ind = (z_qsos < z_qso_cut);
filter_flags(ind) = bitset(filter_flags(ind), 1, true);

% filtering bit 1: BAL
ind = (bal_visual_flags);
filter_flags(ind) = bitset(filter_flags(ind), 2, true);

% filtering bit 4: ZWARNING
ind = (zwarning > 0);
%% but include `MANY_OUTLIERS` in our samples (bit: 1000)
ind_many_outliers      = (zwarning == bin2dec('10000'));
ind(ind_many_outliers) = 0;
filter_flags(ind) = bitset(filter_flags(ind), 5, true);

los_inds = containers.Map();
dla_inds = containers.Map();
z_dlas   = containers.Map();
log_nhis = containers.Map();

% load available DLA catalogs
for catalog_name = {'dr12q_gp'}
  % [train_dr12q] train_ind here to simplicity; move to other file in the future
  % train all of the spectra in dr12q with p_no_dlas > 0.9
  training_release = 'dr12q';
  processed = load(sprintf('%s/processed_qsos_multi_lyseries_a03_zwarn_occams_trunc_dr12q', ...
      processed_directory(training_release)));

  % [test_ind] get all unique IDs from test data from DR12Q
  test_ind = processed.test_ind;
  test_unique_ids_dr12q = unique_ids_dr12q(test_ind); 

  % determine lines of sight searched in this catalog
  los_inds(catalog_name{:}) = ismember(unique_ids, test_unique_ids_dr12q);

  % [p_dlas] determine DLAs by setting threshold on pDLAs
  ind = (processed.p_dlas > 0.9);
  dla_unique_ids = unique_ids_dr12q(test_ind);
  dla_unique_ids = dla_unique_ids(ind);
  % DLA paramteres
  processed_z_dlas   = processed.MAP_z_dlas(ind, 1, 1);
  processed_log_nhis = processed.MAP_log_nhis(ind, 1, 1);

  % determine DLAs flagged in this catalog
  [dla_inds(catalog_name{:}), ind] = ismember(unique_ids, dla_unique_ids);
  ind = find(ind);

  % determine lists of DLA parameters for identified DLAs, when
  % available
  this_z_dlas   = cell(num_quasars, 1);
  this_log_nhis = cell(num_quasars, 1);
  for i = 1:numel(ind)
    this_dla_ind = (dla_unique_ids == unique_ids(ind(i)));
    this_z_dlas{ind(i)}   = processed_z_dlas(this_dla_ind);
    this_log_nhis{ind(i)} = processed_log_nhis(this_dla_ind);
  end
  z_dlas(  catalog_name{:}) = this_z_dlas;
  log_nhis(catalog_name{:}) = this_log_nhis;

end

% save catalog
release = 'dr16q';
variables_to_save = {'sdss_names', 'ras', 'decs', 'thing_ids', 'plates', ...
                     'mjds', 'fiber_ids', 'z_qsos', ...  %'snrs', ...
                     'bal_visual_flags', 'filter_flags', ...
                     'los_inds', 'dla_inds', 'z_dlas', 'log_nhis', ...
                     'zwarning', 'in_dr12', 'in_dr16', 'z_qso_diff_in_dr12', ...
                     'unique_ids', 'unique_ids_dr12q', 'in_dr12_z'};
save(sprintf('%s/catalog', processed_directory(release)), ...
    variables_to_save{:}, '-v7.3');

% build file list for SDSS DR16Q spectra to download (i.e., the ones
% that are not yet removed from the catalog according to the filtering
% flags)
fid = fopen(sprintf('%s/file_list', spectra_directory(release)), 'w');
for i = 1:num_quasars
  if (filter_flags(i) > 0)
    continue;
  end

  % only 5.13.0 here
  fprintf(fid, 'v5_13_0/spectra/lite/./%i/spec-%i-%i-%04i.fits\n', ...
          plates(i), plates(i), mjds(i), fiber_ids(i));

end
fclose(fid);
