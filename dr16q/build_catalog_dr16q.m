% build_catalogs_dr16q: loads existing QSO and DLA catalogs, applies some
% initial filters, and creates a list of spectra to download from SDSS
%
% ZWARNING: ensure we exclude those spectra with bad redshift status reported

% load QSO catalogs
release = 'dr9q';
dr9_catalog = ...
    fitsread(sprintf('%s/DR9Q.fits', distfiles_directory(release)), ...
             'binarytable');

release = 'dr10q';
dr10_catalog = ...
    fitsread(sprintf('%s/DR10Q_v2.fits', distfiles_directory(release)), ...
             'binarytable');

release = 'dr12q';
dr12_catalog = ...
    fitsread(sprintf('%s/DR12Q.fits', distfiles_directory(release)), ...
             'binarytable');

release = 'dr16q';
dr16_catalog = ...
  fitsread(sprintf('%s/DR16Q_v4.fits', distfiles_directory(release)), ...
              'binarytable');

% extract basic QSO information from DR12Q catalog
sdss_names       =  dr16_catalog{1};
ras              =  dr16_catalog{2};
decs             =  dr16_catalog{3};
thing_ids        =  dr16_catalog{16};
plates           =  dr16_catalog{4};
mjds             =  dr16_catalog{5};
fiber_ids        =  dr16_catalog{6};
z_qsos           =  dr16_catalog{27}; % Best available redshift taken from Z VI, Z PIPE, Z DR12Q, Z DR7Q SCH, or Z DR6Q HW
zwarning         =  dr16_catalog{30};
% snrs             =  dr16_catalog{46}; % no SNR in dr16q
bal_visual_flags = (dr16_catalog{57} > 0.75); % 99,856 spectra with BAL PROB ≥ 0.75

num_quasars = numel(z_qsos);

% determine which objects in DR12Q are in DR10Q and DR9Q, using SDSS
% thing IDs
in_dr9  = ismember(thing_ids,  dr9_catalog{4});
in_dr10 = ismember(thing_ids, dr10_catalog{4});
in_dr12 = ismember(thing_ids, dr12_catalog{4});

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
for catalog_name = {'dr9q_concordance', 'dr12q_noterdaeme', 'dr12q_visual'}

  % determine lines of sight searched in this catalog
  los_catalog = ...
      load(sprintf('%s/los_catalog', dla_catalog_directory(catalog_name{:})));
  los_inds(catalog_name{:}) = ismember(thing_ids, los_catalog);

  dla_catalog = ...
      load(sprintf('%s/dla_catalog', dla_catalog_directory(catalog_name{:})));

  % determine DLAs flagged in this catalog
  [dla_inds(catalog_name{:}), ind] = ismember(thing_ids, dla_catalog(:, 1));
  ind = find(ind);

  % determine lists of DLA parameters for identified DLAs, when
  % available
  this_z_dlas   = cell(num_quasars, 1);
  this_log_nhis = cell(num_quasars, 1);
  for i = 1:numel(ind)
    this_dla_ind = (dla_catalog(:, 1) == thing_ids(ind(i)));
    this_z_dlas{ind(i)}   = dla_catalog(this_dla_ind, 2);
    this_log_nhis{ind(i)} = dla_catalog(this_dla_ind, 3);
  end
  z_dlas(  catalog_name{:}) = this_z_dlas;
  log_nhis(catalog_name{:}) = this_log_nhis;

end

% save catalog
release = 'dr16q';
variables_to_save = {'sdss_names', 'ras', 'decs', 'thing_ids', 'plates', ...
                     'mjds', 'fiber_ids', 'z_qsos', ...  %'snrs', ...
                     'bal_visual_flags', 'in_dr9', 'in_dr10', 'filter_flags', ...
                     'los_inds', 'dla_inds', 'z_dlas', 'log_nhis', ...
                     'zwarning', 'in_dr12'};
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
