%--------------------------------------------------------------------------
% Presentation script for vision-touch decoding study
% Th experiment is made up of 3 consecutive sessions (run script 3x)
% Written by Sophie Smit & Denise Moerel - May 2022
%--------------------------------------------------------------------------

%% Clear all old variables, etc., close windows, clear command window 
clearvars; close all; clc;

%% Get experiment directory
[root_folder,exp_name] = fileparts(mfilename('fullpath'));
if isempty(root_folder)       
    root_folder = uigetdir(pwd,'please select the experiment folder');
end
% Add the experiment directory to matlab path
addpath(genpath(root_folder));
% Check if there's a data folder. If not, we make one. 
if exist([root_folder,filesep,'Data'],'dir') ~= 7
    mkdir(root_folder,'Data') 
end

%% Setup Psychtoolbox (if we haven't done so already)
try 
    % See if Psychtoolbox is set up
    PsychtoolboxVersion;
catch
    % If not, swith to the Psychtoolbox folder
    cd(fullfile(fileparts(root_folder),'Psychtoolbox'));
    % Setup Psychtoolbox
    SetupPsychtoolbox;
end

% Running on PTB-3? Abort otherwise.
AssertOpenGL; 

% Switch to the experiment folder
cd(root_folder);

%% Set global parameters
global isEEGexperiment

%% Are we using the EEG?
isEEGexperiment = 1;    % Are we using the EEG? 0 = no 1 = yes

%% Do dummy calls to GetSecs, WaitSecs, KbCheck
KbName('UnifyKeyNames');
KbCheck;
WaitSecs(0.1);
GetSecs;

%% Get subject information
participant_number = input('Participant Number: ');
session_number = input('Session Number: ');

% Make a name for the output file containing the subject information
file_name = [num2str(participant_number,'P%02d'),num2str(session_number,'_Session%02d'),'_Touch_Processing_EEG'];

% If so, we ask if it is ok to overwrite the existing file
if exist([root_folder,filesep,'Data' filesep file_name,'.mat'],'file') > 0
    disp('WARNING! File name has already been used!');
    query = input('Do you want to OVERWRITE the existing file?  y or n(default)?','s');
    if query(1) ~= 'y'
        error('Exiting program...');
    end
end

%% save a copy of the experiment code 
exp_code = fileread([exp_name,'.m']);

%% Set random number generator seed
rand_seed = rng('shuffle');

%% Parameters
num_runs = 4;
num_trials = 144;
num_block = 4; % mini blocks within a run
pos_targets_per_block = 1:4;
num_consec_trials_per_mod = 9;
num_consec_trials_per_ori = num_trials/num_block;

% Maximum number of repeats per finger (across trials)
num_max_reps = 4;

% Video parameters
original_num_frames_ST = 73; % total video frames
total_videos = 1:12;
frame_time = 3;                 

% Touch parameters
touch_time = [0.5,(1/120)*frame_time*4]; % Time of the touch in ms [non_target, target]
touch_frame_current = 39;
            
% Set the size the indiviudal video frames are presented at
img_width = 320;
img_height= 320;
movie_size = 320;

% Parameters for the questions               
% Set possible fingers for the question
pos_fingers = [{'thumb'},{'pinky'}];
% Set opossible feeedback to participant
pos_feedback = [{'incorrect'},{'correct'}];

% Colours
background_cl = 0;                  % Black background
text_cl = 255;                      % Text colour
feedback_cl = [255,0,0;0,200,0];    % Feedback colour
fix_cl = 255;                       % Fixation colour

% Fixation
fix_line_size = 10;
fix_line_width = 4;

% Timing
feedback_time = 1;  % Time the feedback is on the screen (after response) in seconds.
ITI_time = 0.8; % Inter-trial_interval

%% Make the target matrix
% Columns refer to
% Column 1: number of touches to thumb
% Column 2: number of touches to pinky
% Column 3: question (thumb (1) or pinky (2))
% Column 4: number of relevant targets
% Column 5: participant's answer
% Column 6: participant's accuracy

% Make a matrix that codes for the targets
target_mat = nan(num_block,7,num_runs);
n_targets = size(pos_targets_per_block,2);
col_1 = sort(repmat(pos_targets_per_block,[1,(num_block*num_runs)/n_targets]));
col_2 = repmat(sort(repmat(pos_targets_per_block,[1,(num_block*num_runs)/n_targets/n_targets])),[1,n_targets]);

% Shuffle columns
shuf_idx = randsample(1:(num_block*num_runs),(num_block*num_runs));

% Add to the matrix
target_mat(:,1,:) = reshape(col_1(shuf_idx),[num_block,1,num_runs]);
target_mat(:,2,:) = reshape(col_2(shuf_idx),[num_block,1,num_runs]);

% Balance questions to make sure each answer (number of relevant targets)
% is equally likely
balance_answers = 1;
while balance_answers
    % Column 3: question (thumb (1) or pinky (2))
    col_3 = Shuffle(repmat(1:2,[1,(num_block*num_runs)/2]));
    target_mat(:,3,:) = reshape(col_3(shuf_idx),[num_block,1,num_runs]);

    % Column 4: number of relevant targets (correct answer)
    for run_num = 1:num_runs
        for block_num = 1:num_block
            target_mat(block_num,4,run_num) = target_mat(block_num,target_mat(block_num,3,run_num),run_num);
        end
    end
    
    % Check if each answer is balanced
    ans_prob = tabulate(reshape(target_mat(:,4,:),[num_block*num_runs,1]));
    if all(ans_prob(:,2)==(num_block*num_runs)/n_targets)
        balance_answers = 0;
    end
end

%% Make the trial structure for the experiment
% Columns refer to:
% Column 1: modality (1 = visual, 2 = tactile)
% Column 2: hand orientation (palm up (front) = 1, palm down (back) = 2)
% Column 3: finger (thumb = 1, pinky = 2)
% Column 4: touch side (left = 1, right = 2)
% Column 5: movie number (for movies only)
% Column 6: target? (1 = no target, 2 = target)
% Column 7: run number
% Column 8: mini block number
% Column 9: trial number per run
% Column 10: trial number per mini block

% Preallocate the matrix
trial_mat = nan(num_trials+max(pos_targets_per_block)*num_block,10,num_runs);

% Videos
% Single touch
finger = [4 5 6 10 11 12;1 2 3 7 8 9]; % [thumb;pinky]
hand_front_back = [7, 8, 9, 10, 11, 12;1, 2, 3, 4, 5, 6]; % [palm up (front);palm down (back)]
% target
finger_target = [16 17 18 22 23 24;13 14 15 19 20 21]; % [thumb;pinky]
hand_front_back_target = [19, 20, 21, 22, 23, 24;13, 14, 15, 16, 17, 18]; % [palm up (front);palm down (back)]

%determine run order
load(fullfile(root_folder,'run_order.mat'));
if session_number>3
    error(['Could not find the run order for session number ',num2str(session_number)]);
end
if ismember(participant_number,run_order(:,1))
    run_order = reshape(run_order(run_order(:,1)==participant_number,2:end),[4,3])';
    run_order = run_order(session_number,:);
else
    error('Could not find the run order for this participant. Make sure the participant number is added to the ''run_order.mat'' file.');
end

mod_and_ori = [1,2,1,2;1,1,2,2];

for run_num = 1:num_runs

    % Make a temporary matrix
    temp_trial_mat = zeros(num_trials,10);

    % Column 1: modality (1 = visual, 2 = tactile)
    temp_trial_mat(:,1) = repmat(repelem([mod_and_ori(1,run_order(run_num)),find(~ismember(1:2,mod_and_ori(1,run_order(run_num))))],[num_consec_trials_per_mod,num_consec_trials_per_mod]),[1,num_trials/num_consec_trials_per_mod/2]);

    % Column 2: hand orientation (palm up (front) = 1, palm down (back) = 2)
    temp_trial_mat(:,2) = repmat(repelem([mod_and_ori(2,run_order(run_num)),find(~ismember(1:2,mod_and_ori(2,run_order(run_num))))],[num_consec_trials_per_ori,num_consec_trials_per_ori]),[1,num_trials/num_consec_trials_per_ori/2]);

    % Column 3: finger (thumb = 1, pinky = 2)    
    for modality = 1:2
        for ori = 1:2
            get_finger_order = 1;
            while get_finger_order
                % Shuffle the order
                finger_order = Shuffle(repmat(1:2,[1,num_trials/2/2/2]));

                % See what the consecutive trial types are
                f_ord = reshape(finger_order,[9,num_block]);
                f_ord_idx = ((f_ord(2:end,:)-f_ord(1:end-1,:))+f_ord(2:end,:))+1;
                f_ord_info = tabulate(f_ord_idx(:));

                % Check we don't have too many repeats of a finger
                f = reshape([f_ord;zeros(1,4)],[num_block*10,1])';
                i = find(diff(f));
                n = [i numel(f)] - [0 i];
                c = arrayfun(@(X) X-1:-1:0, n , 'un',0);
                max_rep = max(cat(2,c{:}))+1;

                % Check we have balanced trial type orders
                if all(f_ord_info(:,2)==(numel(f_ord_idx)/4)) && max_rep<num_max_reps
                    temp_trial_mat(temp_trial_mat(:,1)==modality&temp_trial_mat(:,2)==ori,3) = finger_order;
                    get_finger_order = 0;
                end
            end
        end
    end
    
    % Column 4: touch side (left = 1, right = 2)
    temp_trial_mat(temp_trial_mat(:,2)==1,4) = ~(temp_trial_mat(temp_trial_mat(:,2)==1,3)-1)+1; % Palm up
    temp_trial_mat(temp_trial_mat(:,2)==2,4) = temp_trial_mat(temp_trial_mat(:,2)==2,3); % Palm down

    % Column 5: movie number (for movies only)
    for ori = 1:2
        for finger_idx = 1:2
            pos_movies = intersect(finger(finger_idx,:),hand_front_back(ori,:));
            temp_trial_mat(temp_trial_mat(:,1)==1&temp_trial_mat(:,2)==ori&temp_trial_mat(:,3)==finger_idx,5) = Shuffle(repmat(pos_movies,[1,num_trials/2/2/2/size(pos_movies,2)]));
        end
    end

    % Column 6: target? (1 = non-target, 2 = target)
    temp_trial_mat(:,6) = 1;

    % Column 8: mini block number
    temp_trial_mat(:,8) = sort(repmat(1:num_trials/num_consec_trials_per_ori,[1,num_consec_trials_per_ori]));

    % Add targets
    for block_num = 1:num_block
        
        % Get the trials for this block
        trial_idx = find(temp_trial_mat(:,8)==block_num);

        % Determine the number of targets at random
        num_targets = target_mat(block_num,1:2,run_num);
        
        % Get the modalities
        target_finger = Shuffle([ones(1,num_targets(1)),ones(1,num_targets(2))*2]);

        % Get the target trial number 
        make_target_idx = 1;
        while make_target_idx
            % Make a random target index
            target_idx = sort(randsample(1:num_consec_trials_per_ori,sum(num_targets)));
            
            % Check modality
            target_mod = temp_trial_mat(target_idx,1);
            target_mod_dif = abs(sum(target_mod==2)-sum(target_mod==1))/sum(num_targets);
            
            % Check if it's not the first/last 2 and not too close
            % together
            spacing = diff(target_idx);
            if all(target_idx>2)&&all(target_idx<(num_consec_trials_per_ori-1))&&all(spacing>2)&&(target_mod_dif<0.5)
                make_target_idx = 0;
            end
        end

        % Add the target(s)
        for target_num = 1:size(target_idx,2)
            % Get the target
            target = [temp_trial_mat(trial_idx(target_idx(1,target_num)),1:2),target_finger(1,target_num),0,0,2,0,block_num,0,0];

            % Calculate the touch side (column 4)
            if target(1,2)==1 % Palm up
                target(1,4) = ~(target(1,3)-1)+1;
            else % palm down
                target(1,4) = target(1,3);
            end

            % Get the movie number (column 5)
            if target(1,1)==1
                % Get the possible movies
                pos_movies = intersect(finger_target(target(1,3),:),hand_front_back_target(target(1,2),:));
                % Pick one at random
                target(1,5) = randsample(pos_movies,1);
            end

            % Add to the matrix
            temp_trial_mat = [temp_trial_mat(1:trial_idx(target_idx(1,target_num)),:);target;temp_trial_mat(trial_idx(target_idx(1,target_num))+1:end,:)];

            % If there are more targets, shift the ID (because we just
            % made the matrix bigger)
            if target_num<size(target_idx,2)
                target_idx(target_num+1:end) = target_idx(target_num+1:end)+1;
                trial_idx = find(temp_trial_mat(:,8)==block_num);
            end

        end

        % Column 10: trial number per mini block
        temp_trial_mat(temp_trial_mat(:,8)==block_num,10) = 1:sum(temp_trial_mat(:,8)==block_num);

    end % Loop over blocks

    % Add Column 7: run number
    temp_trial_mat(:,7) = run_num;

    % Add Column 9: trial number per run
    temp_trial_mat(:,9) = 1:size(temp_trial_mat,1);
    
    % Add the matrix for the run to the big trial structure matrix
    trial_mat(1:size(temp_trial_mat,1),:,run_num) = temp_trial_mat;


end % Loop over runs

%% prepare one sound wave for 2 different output channels
% Sound parameters
sound_params.samprate=44100;
sound_params.t = 0:2/sound_params.samprate:0.3; %0.5; %sounds 500 ms 0:1/sound_params.samprate:1;

% Create the sound
wave_data = ones(1,1000);
no_sound_zeros=zeros(length(wave_data),1)';
channel_one_sound=[no_sound_zeros; wave_data]; % Channel 1 is silent, Channel 2 has wave_data
channel_two_sound=[wave_data; no_sound_zeros]; % Channel 2 has wave_data, Channel 1 is silent

%% Performs basic initialization of the sound driver
InitializePsychSound;
 
% Open a PortAudio audio device and initialize it.
pamaster = PsychPortAudio('Open', [], [], 0, sound_params.samprate, 2);

% Fill the buffer with the sounds
buffer_channel = [PsychPortAudio('CreateBuffer',[], channel_one_sound),PsychPortAudio('CreateBuffer',[], channel_two_sound)];
PsychPortAudio('FillBuffer', pamaster, buffer_channel(1));
PsychPortAudio('FillBuffer', pamaster, buffer_channel(2));   

%% EEG: open i/o port
if isEEGexperiment
    %create IO64 interface object
    try
        p.ioObj = io64;
        % check the port status
        status = io64(p.ioObj);
    catch e
        status = 1;
        disp(['Failed to open io64: ' e.message])
    end
else
    status = 1;
end
if status == 0
    p.address = hex2dec('DFB8');% 'port address BIOSEMI'
    display(['Functioning parallel port opened at: ' num2str(p.address)])
else
    p.ioObj = [];
    p.address = [];
end

% Set triggers to 0
if isEEGexperiment
    io64(p.ioObj,p.address,0);
end

% Try to open a window
try

    % Could use [win, rect]
	[win, wRect] = Screen('OpenWindow',0,background_cl);

    % Get the window size
    window_width = wRect(3); 
    window_height = wRect(4);

    % Get the size of the on screen window in pixels
    % For help see: Screen WindowSize?
    [screenXpixels, screenYpixels] = Screen('WindowSize', win);

    % Calculate the coordinates of the grey square (in between movies)
    movie_ISI_rect = CenterRect([1,1,movie_size,movie_size],wRect);
    movie_ISI_col = 20;

    % get the centre coordinates of the screen
    x0 = window_width/2; 
    y0 = window_height/2;
    
 	% Set parameters for text
    Screen('TextFont',win,'Arial');
    Screen('TextSize',win,20);
    
	% Get the flip interval
    frame_rate = Screen('NominalFrameRate',win);
    if frame_rate~=120
        error('Please make sure the frame rate is 120Hz!');
    end
    frame_rate = 1/frame_rate;

    HideCursor;
    
    % Calculate the time of a single touch visual trial (so we can keep
    % auditory the same)
    trial_time = original_num_frames_ST*(frame_time*frame_rate);

    %% Load The images BEFORE playing them!
    % To test for memory issue, load fewer frames for each video 
    %(instead of 120, load 4)

    % image_indexes will refer to image texture loaded into memory (GPU?)
    image_indexes = zeros(length(total_videos), original_num_frames_ST); % it's good to first create a structure with 0s so Matlab knows that size

    % Loop over videos
    for video_index = 1:length(total_videos) %run through all the videos to prepare the textures
            
            % get the video name 
            video_folder_directory = fullfile(root_folder,'Stimuli'); % 'C:\Users\Sophie..
            video_name = dir(fullfile(video_folder_directory,[num2str(total_videos(video_index),'%02d'),'*.mp4']));
            
            % Load the video
            vidObj = VideoReader(fullfile(video_folder_directory,video_name.name));
            vid = read(vidObj);
            
          	% Loop over frames
            for video_frame_index = 1 : size(vid,4)
                
                % Screen('MakeTexture'... returns a 'Texture Index'
                if ismember(video_index,7:12)
                    image_indexes(video_index,video_frame_index) = Screen('MakeTexture', win, fliplr(vid(:,:,:,video_frame_index))); % convert image_data to texture
                else
                	image_indexes(video_index,video_frame_index) = Screen('MakeTexture', win, vid(:,:,:,video_frame_index)); % convert image_data to texture
                end
               	CheckEscapeKey();

            end

            % --------------------------------------------
            % Make the target video
            target_vid = cat(4,vid(:,:,:,1:42),vid(:,:,:,60:65),vid(:,:,:,15:-1:1),repmat(vid(:,:,:,1),[1,1,1,10]));

            % Loop over frames
            for video_frame_index = 1 : size(target_vid,4)

                % Screen('MakeTexture'... returns a 'Texture Index'
                if ismember(video_index,7:12)
                    image_indexes(video_index+12,video_frame_index) = Screen('MakeTexture', win, fliplr(target_vid(:,:,:,video_frame_index))); % convert image_data to texture
                else
                    image_indexes(video_index+12,video_frame_index) = Screen('MakeTexture', win, target_vid(:,:,:,video_frame_index)); % convert image_data to texture
                end
                CheckEscapeKey();

            end            

            % --------------------------------------------

            % Show feedback about the progress
            percent_complete = 100 * (video_index - 1) / length(total_videos);
            debug_str = sprintf('Loading stimuli: %0.0f%%', percent_complete);
            DrawFormattedText(win, debug_str, 'center', 'center', text_cl);
            Screen('Flip', win);
            
            % Clear variables
            clear vidObj vid target_vid
    end
    
    %load images for instructions
    image_hand_up = imread('hand_up.jpg');
    image_hand_down = imread('hand_down.jpg');
    image_texture = [Screen('MakeTexture', win, image_hand_up),Screen('MakeTexture', win, image_hand_down)];

    %% Instructions
    %Figure out how to present left-aligned
    instructions_str = sprintf(['INSTRUCTIONS\n\n'...
        'On each trial, you will either see a video of a hand being touched or you will feel a touch to your own hand.\n\n' ...
        'The task is to count (and combine) the short touches you see on the screen and feel on your own hand.\n\n' ...
        'You have to count these for the thumb and pinky separately as the question will only ask you about one.\n\n' ...
        'You will be asked how many you counted every 2 minutes, then start counting again at 1.\n\n\n'...
        ''...
        'IMPORTANT\n\n'...
        'It is VERY important that you keep looking at the white cross in the middle of the screen throughout the experiment,\n\n'...
        'including when you are feeling touch on your hand (and no videos are shown).\n\n'...
        'AND also to sit as still as possible.\n\n\n'...
        'There are 4 runs per session (and three sessions in total). Once you are ready, press the space bar to begin.']);

    DrawFormattedText(win, instructions_str, 'center', 'center', text_cl);
    Screen('Flip', win);

    % wait for space key to (re-)start the experimement
    wait_to_continue = 1;
    while wait_to_continue
        [keyIsDown, pressedSecs, keyCode]= KbCheck;
        if keyIsDown
            if strcmp(KbName(find(keyCode)),{'space'})
                wait_to_continue = 0;
            end
        end
    end    
                
    %% Loop over runs        
    for current_run = 1:num_runs %loop through 8 runs determined above for pt

        % Run start message
        DrawFormattedText(win, ['Start run number ', num2str(current_run), ' by pressing the space bar.'], 'center','center', text_cl);
        block_start = Screen('Flip', win);
        WaitSecs(1);
        wait_to_continue = 1;
        while wait_to_continue
            [keyIsDown, pressedSecs, keyCode]= KbCheck;
            if keyIsDown
                if strcmp(KbName(find(keyCode)),{'space'})
                    wait_to_continue = 0;
                end
            end
        end    
        next_trial_start = GetSecs()+0.5;

        %% Loop over trials
        % The number of trials per block varies depending on the number of
        % targets
        for trial_number = 1:max(trial_mat(:,9,current_run))
            
            % report blocks and display text after 32 trials
            if trial_mat(trial_number,10,current_run)==1  
                % Show instruction about the hand orientation
                Screen('DrawTexture', win, image_texture(1,trial_mat(trial_number,2,current_run)), [],[x0-400, y0-200, x0+400, y0+200]);
                DrawFormattedText(win, 'Flip hand and once ready, continue experiment with the space bar' , 'center', y0-600);
                Screen('Flip', win);
                WaitSecs(1);

                %wait for space key to (re-)start the experimement
                wait_to_continue = 1;
                while wait_to_continue
                    [keyIsDown, pressedSecs, keyCode]= KbCheck;
                    if keyIsDown
                        if strcmp(KbName(find(keyCode)),{'space'})
                            wait_to_continue = 0;
                        end
                    end
                end    
    
                % Show fixation
                Screen('DrawLines',win,[[x0-fix_line_size-1;y0],[x0+fix_line_size+1;y0],[x0;y0-fix_line_size-1],[x0;y0+fix_line_size+1]],fix_line_width+2,0);
                Screen('DrawLines',win,[[x0-fix_line_size;y0],[x0+fix_line_size;y0],[x0;y0-fix_line_size],[x0;y0+fix_line_size]],fix_line_width,fix_cl);
                Screen('Flip',win);
                WaitSecs(1);

            end 

            % Get the trial type (movie (1) or touch (2))
            if trial_mat(trial_number,1,current_run)==1 % If this is a 'movie' trial

                % Get the movie number
                video_index = trial_mat(trial_number,5,current_run); %get the number of the movie we want to play

                % Loop over frames
                next_frame_start = next_trial_start;
                num_frames = sum(image_indexes(video_index,:)>0);
                for frame_num = 1:num_frames

                    % Draw the frame
                    Screen('DrawTexture',win,image_indexes(video_index,frame_num),[],[x0-img_width/2, y0-img_height/2, x0+img_width/2, y0+img_height/2]);
                    Screen('DrawLines',win,[[x0-fix_line_size-1;y0],[x0+fix_line_size+1;y0],[x0;y0-fix_line_size-1],[x0;y0+fix_line_size+1]],fix_line_width+2,0);
                    Screen('DrawLines',win,[[x0-fix_line_size;y0],[x0+fix_line_size;y0],[x0;y0-fix_line_size],[x0;y0+fix_line_size]],fix_line_width,fix_cl);
                    current_frame(trial_number,frame_num,current_run) = Screen('Flip',win,next_frame_start);

                    % Send trigger
                    if isEEGexperiment
                        if frame_num==1
                            io64(p.ioObj,p.address,4);
                        elseif frame_num==touch_frame_current
                            io64(p.ioObj,p.address,8);
                        else
                            io64(p.ioObj,p.address,0);
                        end
                    end

                % Calculate the onset of the next frame
                next_frame_start = current_frame(trial_number,frame_num,current_run) + (frame_time*frame_rate);

                end % end frames

                % Calculate ISI start time
                isi_start = next_frame_start;
                
                % Check for ecs key
                CheckEscapeKey();

           else % If this is a 'touch' trial                           

                % Show fixation
                Screen('DrawLines',win,[[x0-fix_line_size;y0],[x0+fix_line_size;y0],[x0;y0-fix_line_size],[x0;y0+fix_line_size]],fix_line_width,fix_cl);
                Screen('Flip',win);
                if isEEGexperiment
                    io64(p.ioObj,p.address,4);
                    WaitSecs(0.01);
                    io64(p.ioObj,p.address,0);
                end         
                
                % Start the touch
                Screen('DrawLines',win,[[x0-fix_line_size;y0],[x0+fix_line_size;y0],[x0;y0-fix_line_size],[x0;y0+fix_line_size]],fix_line_width,fix_cl);
                current_frame(trial_number,1,current_run) = Screen('Flip',win,next_trial_start);
                if isEEGexperiment
                 	io64(p.ioObj,p.address,trial_mat(trial_number,3,current_run));
                    WaitSecs(touch_time(trial_mat(trial_number,6,current_run)));
                    io64(p.ioObj,p.address,0);
                end               

            	% Calculate ISI start time
                isi_start = current_frame(trial_number,1,current_run)+trial_time;
    
                CheckEscapeKey();

            end % end trial

            %% Inter-trial interval
            % Show empty screen
            if trial_mat(trial_number,1,current_run)==1 % If this is a 'movie' trial
                Screen('FillRect',win,movie_ISI_col,movie_ISI_rect);
            end
            Screen('DrawLines',win,[[x0-fix_line_size-1;y0],[x0+fix_line_size+1;y0],[x0;y0-fix_line_size-1],[x0;y0+fix_line_size+1]],fix_line_width+2,0);
            Screen('DrawLines',win,[[x0-fix_line_size;y0],[x0+fix_line_size;y0],[x0;y0-fix_line_size],[x0;y0+fix_line_size]],fix_line_width,fix_cl);

            % Calculate the onset of the next trial
            trial_end = Screen('Flip',win,isi_start);
            next_trial_start = trial_end+ITI_time;
            
            %% Ask question after each mini block
            if trial_number==max(trial_mat(:,9,current_run))||trial_mat(trial_number,10,current_run)>trial_mat(trial_number+1,10,current_run)
                % Get the number of the little block
                block_num = trial_mat(trial_number,8,current_run);

                % Ask the question
                Screen('Fillrect',win,255);
                DrawFormattedText(win,['How many targets (short touches) were there to the ',pos_fingers{target_mat(block_num,3,current_run)},'?'], 'center','center', 0);
                q_on = Screen('Flip', win);
                
                % Get the response
                get_resp = 1;
                while get_resp
                	[keyIsDown, pressedSecs, keyCode]= KbCheck;
                    if keyIsDown
                        resp = str2double(KbName(find(keyCode)));
                        % Check if they pressed a number
                        if ismember(resp,0:9)
                            rt = pressedSecs-q_on;
                        	get_resp = 0;
                        end
                    end
                end

                % Save the response
                target_mat(block_num,5,current_run) = resp;
                
                % Save the RT
                target_mat(block_num,6,current_run) = rt;

                % Calculate the accuracy (0 = incorrect, 1 = correct )
                target_mat(block_num,7,current_run) = target_mat(block_num,4,current_run)==resp;

                % Give feedback to participant
                Screen('Fillrect',win,0);
                DrawFormattedText(win, ['Your response is ' pos_feedback{target_mat(block_num,7,current_run)+1}], 'center',y0-40, feedback_cl(target_mat(block_num,7,current_run)+1,:));
                DrawFormattedText(win, ['The number of targets to the ',pos_fingers{target_mat(block_num,3,current_run)},' was ' num2str(target_mat(block_num,4,current_run)) '. \n\n Press the space bar to continue'], 'center',y0+40, text_cl);
                Screen('Flip', win);
                WaitSecs(0.5);
                
                % save data after every sequence
                % touch_space % touch_finger %hand_orientation
                save([root_folder,filesep,'Data',filesep,file_name]);

                % Calculate the onset of the next trial
                wait_to_continue = 1;
                while wait_to_continue
                	[keyIsDown, pressedSecs, keyCode]= KbCheck;
                    if keyIsDown
                        if strcmp(KbName(find(keyCode)),{'space'})
                        	wait_to_continue = 0;
                        end
                    end
                end       
            end


        end % Loop over trials - end trial matrix

    end %end num_runs loop - end of experiment (8 runs)

    DrawFormattedText(win, 'The end of this part', 'center', 'center', text_cl);
    Screen('Flip', win);
    WaitSecs(2);


    catch err % if error occurs
    
        % save entire workspace
        save([root_folder,filesep,'Data',filesep,file_name]);

        % Close the screen
        Screen('CloseAll');
        
        % Output the error message that describes the error:
        rethrow(err);
    
end

Screen('CloseAll');

%% Functions

% use escape to quit
function CheckEscapeKey()
    [keyIsDown, ~, keyCode, ~] = KbCheck;

    %end the experiment with esc
    if keyIsDown==1 && strcmp(KbName(find(keyCode)),'esc')
        error('Esc pressed to ESCAPE');
    end
    close all 
end





