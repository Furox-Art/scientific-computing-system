%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 4c - plot between-modality decoding difference between groups
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: cbrewer, matplotlib

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_bids';

for samples = 1:2 % different samples (VT and no VT)
    
    %% Set the participant number
    name = {'VT_group' 'VT_group'};
    VT_group = [8	9	12	13	14	16	18	20	21	23	28	29	30	31	32	33	35	36	38	39	40];
    NoVT_group = [1	2	3	4	5	6	7	10	11  15	17	19	22	24  25	26	27	34	37];
    
    if samples == 1
        subs_all = VT_group;
    else
        subs_all = NoVT_group;
    end
    
    subs_excluded = [10,13,16,18,32,33]; %excluded based on task performance <60%
    sub_ids = setdiff(subs_all,subs_excluded);
    num_subs = size(sub_ids,2);
    
    %% Calculate Bayes factors
    timegen_data_touch2vis = zeros(744,744,num_subs);
    
    %% Load the data
    % loop over participats
    for sub_idx = 1:num_subs
        
        %Get the participant ID
        sub_num = sub_ids(1,sub_idx);
        fprintf('Loading data S%.0f of %.0f\n',sub_idx,num_subs);
        load(fullfile(current_dir,'derivatives','crossdecoding',[num2str(sub_num,'S%02d'),'_task-touchdecoding_crossdecoding_SL.mat']));
        
        % Reshape the data
        [data1, labels, values] = cosmo_unflatten(res_timegen_touch2vis,1);
        timegen_data_touch2vis(:,:,sub_idx) = data1;
    end
    
    if samples == 1
        timegen_data_touch2vis1 = timegen_data_touch2vis;
    else
        timegen_data_touch2vis2 = timegen_data_touch2vis;
    end
    
end

%% Calculate Bayes Factors

% Get the BF for each timepoint x timepoint combination by looping over the
% test times for each train time (across participants - using squeeze).
% %mu = 0 cause H0 is 0, the null interval is 0.5

% Calculate Bayes Factors between groups
bf_BetweenGroups = zeros(size(timegen_data_touch2vis,1),size(timegen_data_touch2vis,2));
for test_time = 1:size(timegen_data_touch2vis,2)
    bf_BetweenGroups(:,test_time) = bayesfactor_R_wrapper_2sample(squeeze(timegen_data_touch2vis1(:,test_time,:)),...
        squeeze(timegen_data_touch2vis2(:,test_time,:)),'args','mu=0,rscale="medium",nullInterval=c(0.5,Inf)');
end

% Save the BFs (because they take a long time to calculate)
save(fullfile(current_dir,'bayes_factors_groups.mat'), 'bf_BetweenGroups');

%% create BF step values
A = bf_BetweenGroups;

% H1
A(A>1 & A<3)        = 1;
A(A>3 & A<6)        = 2;
A(A>6 & A<10)       = 3;
A(A>10)             = 4;
% H0
A(A>1/3 & A<1)      = 5;
A(A<1/3 & A>1/6)    = 6;
A(A<1/6 & A>1/10)   = 7;
A(A<1/10)           = 8;

C = zeros(size(A));

% have to change the numbers so we can plot with BF increasing and
% decreasing (above is done in a specific way so it doesn't change the
% numbers over and over)
C(A==8) = 1; C(A==7) = 2; C(A==6) = 3; C(A==5) = 4;
C(A==1) = 5; C(A==2) = 6; C(A==3) = 7; C(A==4) = 8;

bf_BetweenGroups_steps = C;


%% Figure specifications
font_size = 26;
titles = {''};
title_letter = {'Tactile to visual: group difference'};
plot_fn = {'Timegen_touch2vis_group_difference'};

% define colours: red - blue
cm2 =  [1.0000    1.0000    1.0000
        0.9451    0.9451    0.9451
        0.8941    0.8941    0.8941
        0.8431    0.8431    0.8431
        0.9922    0.8588    0.7804
        0.9569    0.6471    0.5098
        0.8392    0.3765    0.3020
        0.6980    0.0941    0.1686];

% Get the timescale labels
num_timepoints = 744;
tv = values{1};
t_idx = 27:64:num_timepoints;
t_label = round(tv(t_idx));

% location of the plots in the figure
ypos = [0.57,0.11];
xpos = [0.071,0.56];

% define range
cb_lim = [50,52;1,8];

%% plot

f = figure(1); clf;
f.Position=[f.Position(1:2) 1500 1500];
f.PaperPositionMode='auto';
f.Resize='off';

% Create the axes
ax1 = axes('Position',[xpos(1),ypos(1),0.32,0.32],'LineWidth',1.5);
ax1.Clipping = 'off';
hold on

% Plot
imagesc(bf_BetweenGroups_steps(:,:,:),cb_lim(2,:));

% Set the colormap
colormap(ax1,cm2);

% Grid line at 0ms
line([find(tv==0),find(tv==0)],[1,num_timepoints+1],'LineWidth',1.5,'Color','k');
line([find(tv==-1000),find(tv==-1000)],[1,num_timepoints+1],'LineWidth',1.5,'Color','k');
text(45,278,'Movie onset','FontSize',font_size,'HorizontalAlignment','right','Rotation',90);
text(301,278,'Touch onset','FontSize',font_size,'HorizontalAlignment','right','Rotation',90);
text(715,301,'Touch onset','FontSize',font_size,'HorizontalAlignment','right');
line([1,num_timepoints+1],[find(tv==0),find(tv==0)],'LineWidth',1.5,'Color','k');
line([1,num_timepoints+1],[find(tv==-1000),find(tv==-1000)],'LineWidth',1.5,'Color','k');
% Diagonal
line([1,num_timepoints+1],[1,num_timepoints+1],'LineWidth',1.5,'Color','k');

% Colorbar
cb = colorbar;
cb.Position = [xpos(1)+0.34,ypos(1),0.008,0.32];
cb.LineWidth = 1;
cb.TickLabels = [{'<1/10'},{'1/10 - 1/6'},{'1/6 - 1/3'},{'1/3 - 1'},{'1 - 3'},{'3 - 6'},{'6 - 10'},{'>10'}];
% the bar goes from 1 - 8 so 7 steps (7/16 = 0.4375) first step is 1.4375, second step is 1.4375 + 2*.4375 etc.
cb.XTick = [1.4375,2.3135,3.1875,4.0625,4.9375,5.8125,6.6875,7.5625];

% Annotate the colorbar

% Annotate evidence levels for the BFs
% Evidence for Ha
ha = annotation('line','Color','k','LineWidth',1.5);
ha.Parent = ax1;
ha.X = [num_timepoints+25,num_timepoints+60];
ha.Y = [(num_timepoints/2),(num_timepoints/2)];
ha = annotation('arrow','Color','k','LineWidth',1.5,'HeadStyle','vback1');
ha.Parent = ax1;
ha.X = [num_timepoints+25,num_timepoints+25];
ha.Y = [(num_timepoints/2),(num_timepoints/2)+40];
text(num_timepoints+25,(num_timepoints/2)+45,'Evidence for Ha','Rotation',90,'FontSize',font_size);

% Evidence for H0
ha = annotation('arrow','Color','k','LineWidth',1.5,'HeadStyle','vback1');
ha.Parent = ax1;
ha.X = [num_timepoints+25,num_timepoints+25];
ha.Y = [(num_timepoints/2),(num_timepoints/2)-40];
text(num_timepoints+25,(num_timepoints/2)-45,'Evidence for H0','Rotation',90,'FontSize',font_size,'HorizontalAlignment','right');
text(num_timepoints+250,num_timepoints/2,'Bayes factors','Rotation',90,'FontSize',font_size,'HorizontalAlignment','center'); %130

% set this before specifying font sizes otherwise they will be reset
set(gca,'FontSize',font_size);

% Set the axes
ax = gca;
set(ax,'Layer','top')
ylabel('Training time (ms) - tactile trials', 'FontSize', font_size);
xlabel('Testing time (ms) - visual trials', 'FontSize', font_size);

xlim([1,num_timepoints+1]);
ylim([1,num_timepoints+1]);
ax.YTick = t_idx;
ax.YTickLabels = t_label;
ax.XTick = t_idx;
ax.XTickLabels = t_label;
ax.XTickLabelRotation = 45;
box on

% Title
text(-160,830,title_letter,'FontSize',font_size+8,'HorizontalAlignment','left','FontWeight','bold'); % letter + sample


%% Save plot
fn = fullfile(current_dir,'plots',plot_fn{1});
tn = tempname;
print(gcf,'-dpng','-r500',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=1;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');


