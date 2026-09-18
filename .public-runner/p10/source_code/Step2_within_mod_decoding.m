%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 2 - within-modality decoding
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

res_searchlight = cell(1,2);

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
    
    %% Baseline correction
    EEG_epoch = pop_rmbase(EEG_epoch,[],1:26); % 26 timepoints (=100ms)
    
    %% Convert to cosmo
    ds = cosmo_flatten(permute(EEG_epoch.data,[3 1 2]),{'chan','time'},{{EEG_epoch.chanlocs.labels},EEG_epoch.times},2);
    ds.a.meeg = struct(); % or cosmo thinks it's not a meeg ds
    ds.sa = table2struct(event_list_table_all_sessions,'ToScalar',true);
    cosmo_check_dataset(ds,'meeg'); % check if it's in the right format
    
    % NOTE: biosemi has label 'Afz', but fieldtrip has label 'AFz'
    ds.a.fdim.values{1}(strcmp(ds.a.fdim.values{1},'Afz')) = {'AFz'};
    
    %% Remove double touch (target) trials from analysis
    ds = cosmo_slice(ds,ds.sa.target==1,1);
    
    %% WITHIN MODALITY DECODING
    dec_acc_som = zeros(2,size(ds.a.fdim.values{2},2)); % touched finger
    
    % Set random number generator
    rng(1);
    
    % loop over vision and touch
    for modality = 1:2
        
        %% Classification: Finger
        
        % Select the data
        ds_sel = cosmo_slice(ds,ds.sa.modality==modality,1);
        
        % How many folds do we want?
        num_folds = size(ds_sel.samples,1)/4; % leave 4 out, one of each option (thumb v pinky; hand up v hand down)
        
        % Define the 4 options for touch side/finger combo
        a = find(ds_sel.sa.hand_ori == 1 & ds_sel.sa.finger == 1);
        b = find(ds_sel.sa.hand_ori == 2 & ds_sel.sa.finger == 1);
        c = find(ds_sel.sa.hand_ori == 1 & ds_sel.sa.finger == 2);
        d = find(ds_sel.sa.hand_ori == 2 & ds_sel.sa.finger == 2);
        
        % Shuffle the trial numbers for each condition
        a = randsample(a,size(a,1));
        b = randsample(b,size(b,1));
        c = randsample(c,size(c,1));
        d = randsample(d,size(d,1));
        
        % Split the shuffled trial numbers into the n folds (for each condition)
        a_reshape = reshape(a,[size(a,1)/num_folds,num_folds]);
        b_reshape = reshape(b,[size(b,1)/num_folds,num_folds]);
        c_reshape = reshape(c,[size(c,1)/num_folds,num_folds]);
        d_reshape = reshape(d,[size(d,1)/num_folds,num_folds]);
        
        % Combine the trial numbers across conditions
        % Now we have the trial numbers for each fold, counterbalanced
        % across the different conditions
        combined = [a_reshape;b_reshape;c_reshape;d_reshape];
        
        % Now we make the partitions
        % Preallocate
        ma.partitions = [];
        ds_sel.sa.chunks = zeros(size(ds_sel.samples,1),1);
        % Loop over folds
        for fold = 1:num_folds
            % Pick the test indices
            ma.partitions.test_indices{fold} = combined(:,fold);
            % Pick the train indices (all the other trials)
            train = combined(:,~ismember(1:num_folds,fold));
            ma.partitions.train_indices{fold} = train(:);
            % Save the chunks
            ds_sel.sa.chunks(ma.partitions.test_indices{fold}) = fold;
        end
        
        % Select targets
        ds_sel.sa.targets = ds_sel.sa.finger;
        
        % Set classification parameters
        ma.classifier = @cosmo_classify_lda; % Classifier type: LDA
        ma.nproc = nproc;
        
        % compute neighborhoods stretching intervals
        nh = cosmo_interval_neighborhood(ds_sel,'time','radius',0);
        
        % Run the classifier
        res = cosmo_searchlight(ds_sel,nh,@cosmo_crossvalidation_measure,ma);
        
        dec_acc_som(modality,:) = res.samples;
        
        %%  Decoding using a searchlight
        
        % Make the neighbourhoors
        nh1 = cosmo_meeg_chan_neighborhood(ds_sel,'count',4,'label','dataset','label_threshold',.99);
        nh2 = cosmo_interval_neighborhood(ds_sel,'time','radius',0);
        nh = cosmo_cross_neighborhood(ds_sel,{nh1,nh2});
        
        % Set classification parameters
        ma.classifier = @cosmo_classify_lda; % Classifier type: LDA
        ma.nproc = nproc;
        
        % Run the classifier
        res_searchlight{modality} = cosmo_searchlight(ds_sel,nh,@cosmo_crossvalidation_measure,ma);
        
    end
    
    %% Save the decoding
    save(fullfile(current_dir,'derivatives/decoding',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_decoding.mat']),'dec_acc_som', 'res_searchlight', 'ds', '-v7.3');
    
end

