%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 3a - cross-decoding (between modalities)
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: eeglab, cosmomvpa

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_bids';

% get number of processes available from Matlab parallel processing pool
nproc = cosmo_parallel_get_nproc_available;

%% Set the participant number
subs_all = 1:40;
subs_excluded = [10,13,16,18,32,33]; %excluded based on task performance <60
sub_ids = setdiff(subs_all,subs_excluded);
num_subs = size(sub_ids,2);

% loop over participants
for sub_idx = 1:num_subs
    
    %% Get the participant ID
    sub_num = sub_ids(1,sub_idx);
    
    %% Load Matlab data: loads event_list_table_all_sessions
    load(sprintf('%s/derivatives/eventslist/sub-%02i_task-touchdecoding_events.mat',current_dir,sub_num));
    
    % Convert coplete events_list to a table (for cosmo)
    event_list_table_all_sessions = array2table(event_list,'VariableNames',{'modality','hand_ori','finger','touch_side','movie_num',...
        'target','run_num','block_num','trial_num_per_run','trial_num_per_block'});
    
    %% Load detrended & epoched EEG data
    EEG_epoch = pop_loadset(sprintf('%s/derivatives/detrended_epoched_data/sub-%02i_task-touchdecoding_eeg.set',current_dir,sub_num));
    
    %% Convert to cosmo
    ds = cosmo_flatten(permute(EEG_epoch.data,[3 1 2]),{'chan','time'},{{EEG_epoch.chanlocs.labels},EEG_epoch.times},2);
    ds.a.meeg = struct(); % or cosmo thinks it's not a meeg ds
    ds.sa = table2struct(event_list_table_all_sessions,'ToScalar',true);
    cosmo_check_dataset(ds,'meeg'); % check if it's in the right format
    
    % NOTE: biosemi has label 'Afz', but fieldtrip has label 'AFz'
    ds.a.fdim.values{1}(strcmp(ds.a.fdim.values{1},'Afz')) = {'AFz'};
    
    % Set random number generator
    rng(1);
    
    %% Remove double touch (target) trials from analysis
    ds = cosmo_slice(ds,ds.sa.target==1,1);
    
    %% Cross decoding: train on touch, test on visual
    % Select the data
    ds_sel = ds;
    
    % Define the chunks (training chunks = 1 & testing chunks = 2)
    ds_sel.sa.chunks = ~((ds_sel.sa.modality)-1)+1;
    
    % Define targets
    ds_sel.sa.targets = ds_sel.sa.finger;
    
    % Set classification parameters
    ma={};
    ma.classifier = @cosmo_classify_lda; % Classifier type: LDA
    ma.measure=@cosmo_crossvalidation_measure;
    ma.dimension='time';
    ma.output='accuracy';
    ma.nproc = nproc;
    
    % Select the data
    ds_sel = cosmo_dim_transpose(ds_sel,'time',1);
    
    % Run the time-gen
    res_timegen_touch2vis = cosmo_dim_generalization_measure(ds_sel,ma);
    
    %% Cross decoding: train on visual, test on touch
    % Select the data
    ds_sel = ds;
    
    % Define the chunks (training chunks = 1 & testing chunks = 2)
    ds_sel.sa.chunks = ds_sel.sa.modality;
    
    % Define targets
    ds_sel.sa.targets = ds_sel.sa.finger;
    
    % Set classification parameters
    ma={};
    ma.classifier = @cosmo_classify_lda; % Classifier type: LDA
    ma.measure=@cosmo_crossvalidation_measure;
    ma.output='accuracy';
    ma.nproc = nproc;
    ma.dimension='time';
    
    % Select the data
    ds_sel = cosmo_dim_transpose(ds_sel,'time',1);
    
    % Run the time-gen
    res_timegen_vis2touch = cosmo_dim_generalization_measure(ds_sel,ma);
    
    %% Save
    save(fullfile(current_dir,'derivatives','crossdecoding',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_crossdecoding.mat']),'res_timegen*', 'ds');
    
end

