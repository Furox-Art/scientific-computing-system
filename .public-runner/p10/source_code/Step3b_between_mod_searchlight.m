%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 3b - cross-decoding (between modalities) using a searchlight
% Note: takes a long time to run
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: eeglab, cosmomvpa

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_test';

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
    
    %% Remove double touch (target) trials from analysis
    % Select the data
    ds = cosmo_slice(ds,ds.sa.target==1,1);
    
    % Set random number generator
    rng(1);
    
    %% DECODING train on touch, test on vision with time-by-time generalization using a searchlight
    % Select the data
    ds_sel = ds;
    
    % Define the chunks (training chunks = 1 & testing chunks = 2)
    ds_sel.sa.chunks = ~((ds_sel.sa.modality)-1)+1;
    
    % Define targets
    ds_sel.sa.targets = ds_sel.sa.finger;
    
    % how many channel locations in each searchlight
    nchannel_locations_searchlight=4;
    
    % make 'time' a sample dimension (necessary for cartesian_dim_transfer)
    ds_time=cosmo_dim_transpose(ds_sel,'time',1);
    
    % EEG as input
    use_chan_type='eeg';
    
    % define searchlight neighborhood
    nbrhood=cosmo_meeg_chan_neighborhood(ds_time,'count',nchannel_locations_searchlight,'chantype',use_chan_type);
    
    % set the measure to be the dim generalization measure
    measure=@cosmo_dim_generalization_measure;
    
    % define the arguments for the measure
    ma=struct();
    ma.classifier = @cosmo_classify_lda; % Classifier type: LDA
    ma.measure=@cosmo_crossvalidation_measure;
    ma.radius=0; %single timepoint
    ma.dimension='time';
    ma.output='accuracy';
    ma.nproc = nproc;
    
    % run transfer across time with the searchlight neighborhood
    cdt_ds=cosmo_searchlight(ds_time,nbrhood,measure,ma);
    
    % move {train,test}_time from being sample dimensions to feature
    % dimensions, so they can be mapped to a fieldtrip struct
    cdt_tf_ds=cosmo_dim_transpose(cdt_ds,{'train_time','test_time'},2);
    
    % trick fieldtrip into thinking this is a time-freq-chan dataset, by
    % renaming train_time and test_time to freq and time, respectively
    cdt_tf_ds=cosmo_dim_rename(cdt_tf_ds,'train_time','freq');
    cdt_tf_ds=cosmo_dim_rename(cdt_tf_ds,'test_time','time');
    
    
    %% Save
    save(fullfile(current_dir,'derivatives','crossdecoding_searchlight',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_crossdecoding_SL.mat']),'cdt_tf_ds', 'cdt_ds');
    
end

