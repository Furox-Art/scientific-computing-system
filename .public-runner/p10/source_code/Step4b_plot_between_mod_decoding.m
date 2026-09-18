%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 4b - plot between-modality decoding
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: cbrewer, matplotlib

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_bids';

for samples = 1%:2 % different samples (VT and no VT)
    
    %% Set the participant number
    name = {'VT_group' 'NoVT_group'};
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
    timegen_data_vis2touch = zeros(744,744,num_subs);
    
    %% Load the data
    % loop over participats
    for sub_idx = 1:num_subs
        
        %Get the participant ID
        sub_num = sub_ids(1,sub_idx);
        fprintf('Loading data S%.0f of %.0f\n',sub_idx,num_subs);
        load(fullfile(current_dir,'derivatives','crossdecoding',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_crossdecoding.mat']));
        
        % Reshape the data
        [data1, labels, values] = cosmo_unflatten(res_timegen_touch2vis,1);
        timegen_data_touch2vis(:,:,sub_idx) = data1;
        data1 = cosmo_unflatten(res_timegen_vis2touch,1);
        timegen_data_vis2touch(:,:,sub_idx) = data1;
    end
    
    %% Calculate Bayes Factors
    
    % Get the BF for each timepoint x timepoint combination by looping over the
    % test times for each train time (across participants - using squeeze).
    % mu = chance, the null interval is 0.5
    
    %%% TOUCH 2 VISUAL
    bf_touch2vis = zeros(size(timegen_data_touch2vis,1),size(timegen_data_touch2vis,2));
    for test_time = 1:size(timegen_data_touch2vis,2)
        bf_touch2vis(:,test_time) = bayesfactor_R_wrapper(squeeze(timegen_data_touch2vis(:,test_time,:)),'args','mu=0.5,rscale="medium",nullInterval=c(0.5,Inf)');
    end
    
    %%% VISUAL 2 TOUCH
    bf_vis2touch = zeros(size(timegen_data_vis2touch,1),size(timegen_data_vis2touch,2));
    for test_time = 1:size(timegen_data_vis2touch,2)
        bf_vis2touch(:,test_time) = bayesfactor_R_wrapper(squeeze(timegen_data_vis2touch(:,test_time,:)),'args','mu=0.5,rscale="medium",nullInterval=c(0.5,Inf)');
    end
    
    % Save the BFs (because they take a long time to calculate)
    save(fullfile(current_dir,'derivatives',['bayes_factors' name{samples} '.mat']), 'bf_touch2vis', 'bf_vis2touch', 'timegen_data_touch2vis', 'timegen_data_vis2touch');
    
end

for samples = 1%:2 % different samples (VT and no VT)
    
    %% Load the BFs
    load(fullfile(current_dir,'derivatives',['bayes_factors' name{samples} '.mat']));
    
    %% create BF step values
    A = bf_touch2vis;
    B = bf_vis2touch;
    
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
    
    % H1
    B(B>1 & B<3)        = 1;
    B(B>3 & B<6)        = 2;
    B(B>6 & B<10)       = 3;
    B(B>10)             = 4;
    % H0
    B(B>1/3 & B<1)      = 5;
    B(B<1/3 & B>1/6)    = 6;
    B(B<1/6 & B>1/10)   = 7;
    B(B<1/10)           = 8;
    
    C = zeros(size(A));
    D = zeros(size(B));
    
    % have to change the numbers so we can plot with BF increasing and
    % decreasing (above is done in a specific way so it doesn't change the
    % numbers over and over)
    C(A==8) = 1; C(A==7) = 2; C(A==6) = 3; C(A==5) = 4;
    C(A==1) = 5; C(A==2) = 6; C(A==3) = 7; C(A==4) = 8;
    
    D(B==8) = 1; D(B==7) = 2; D(B==6) = 3; D(B==5) = 4;
    D(B==1) = 5; D(B==2) = 6; D(B==3) = 7; D(B==4) = 8;
    
    bf_touch2vis_steps  = C;
    bf_vis2touch_steps  = D;
    
    % combine BF
    if samples == 1
        bf_touch2vis_steps1 = bf_touch2vis_steps;
        bf_vis2touch_steps1 = bf_vis2touch_steps;
        timegen_data_touch2vis1 = (mean(timegen_data_touch2vis(:,:,:),3))*100;
        timegen_data_vis2touch1 = (mean(timegen_data_vis2touch(:,:,:),3))*100;
    else
        bf_touch2vis_steps2 = bf_touch2vis_steps;
        bf_vis2touch_steps2 = bf_vis2touch_steps;
        timegen_data_touch2vis2 = (mean(timegen_data_touch2vis(:,:,:),3))*100;
        timegen_data_vis2touch2 = (mean(timegen_data_vis2touch(:,:,:),3))*100;
    end
    
end

% combine sample data
bf_steps_combined = cat(3, bf_touch2vis_steps1, bf_touch2vis_steps2, bf_vis2touch_steps1, bf_vis2touch_steps2);
timegen_data_combined = cat(3, timegen_data_touch2vis1, timegen_data_touch2vis2, timegen_data_vis2touch1, timegen_data_vis2touch2);

%% Figure specifications
font_size = 26;
font_size_legend = 20;
titles = [{'Tactile to visual: decoding accuracy'},{'Tactile to visual: Bayes factors'},{'Visual to tactile: decoding accuracy'},{'Visual to tactile: Bayes factors'}];
title_letter = [{'a Vicarious touch group'},{'b No vicarious touch group'}];
plot_fn = [{'Timegen_touch2vis'},{'Timegen_vis2touch'}];

% define colours
cm1 = tab10; % you need matplotlib for this
cm1 = cm1([1,2,4],:);

% red - blue
cm2 =  [1.0000    1.0000    1.0000
        0.9451    0.9451    0.9451
        0.8941    0.8941    0.8941
        0.8431    0.8431    0.8431
        0.9922    0.8588    0.7804
        0.9569    0.6471    0.5098
        0.8392    0.3765    0.3020
        0.6980    0.0941    0.1686];

% define range
cb_lim = [50,52;1,8];

% get the timescale labels
num_timepoints = 744;
tv = values{1};
t_idx = 27:64:num_timepoints;
t_label = round(tv(t_idx));

% location of the plots in the figure
ypos = [0.57,0.11];
xpos = [0.071,0.56];

%% plot
for crossdecoding_direction = 1:2 %(touch->vis & vis->touch)
    
    f = figure(crossdecoding_direction); clf;
    f.Position=[f.Position(1:2) 1500 1500];
    f.PaperPositionMode='auto';
    f.Resize='off';
    %f.Visible = 'off'; % don't show figure
    
    for plot_type_idx = 1:2 %accuracy/BF
        for invariance_idx = 1%:2 %samples (VT and no VT)
            % Create the axes
            ax1 = axes('Position',[xpos(plot_type_idx),ypos(invariance_idx),0.32,0.32],'LineWidth',1.5);
            ax1.Clipping = 'off';
            hold on
            
            % Plot
            if crossdecoding_direction == 1 % touch 2 vis
                if plot_type_idx==1 % accuracy
                    if invariance_idx==1 % VT_group
                        imagesc(timegen_data_combined(:,:,1),cb_lim(plot_type_idx,:));
                    elseif invariance_idx==2 % NoVT_group
                        imagesc(timegen_data_combined(:,:,2),cb_lim(plot_type_idx,:));
                    end
                elseif plot_type_idx==2 % BFs
                    if invariance_idx==1 % VT_group
                        imagesc(bf_steps_combined(:,:,1),cb_lim(plot_type_idx,:));
                    elseif invariance_idx==2 % NoVT_group
                        imagesc(bf_steps_combined(:,:,2),cb_lim(plot_type_idx,:));
                    end
                end
            elseif crossdecoding_direction == 2 % vis 2 touch
                if plot_type_idx==1 % accuracy
                    if invariance_idx==1 % VT_group
                        imagesc(timegen_data_combined(:,:,3),cb_lim(plot_type_idx,:));
                    elseif invariance_idx==2 % NoVT_group
                        imagesc(timegen_data_combined(:,:,4),cb_lim(plot_type_idx,:));
                    end
                elseif plot_type_idx==2 % BFs
                    if invariance_idx==1 % VT_group
                        imagesc(bf_steps_combined(:,:,3),cb_lim(plot_type_idx,:));
                    elseif invariance_idx==2 % NoVT_group
                        imagesc(bf_steps_combined(:,:,4),cb_lim(plot_type_idx,:));
                    end
                end
            end
            
            % Set the colormap
            if plot_type_idx == 1
                colormap(ax1,[linspace(1,cm1(invariance_idx,1),256);linspace(1,cm1(invariance_idx,2),256);linspace(1,cm1(invariance_idx,3),256)]');
            else
                colormap(ax1,cm2);
            end
            
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
            cb.Position = [xpos(plot_type_idx)+0.34,ypos(invariance_idx),0.008,0.32];
            cb.LineWidth = 1;
            if plot_type_idx==2
                cb.TickLabels = [{'<1/10'},{'1/10 - 1/6'},{'1/6 - 1/3'},{'1/3 - 1'},{'1 - 3'},{'3 - 6'},{'6 - 10'},{'>10'}];
                % the bar goes from 1 - 8 so 7 steps (7/16 = 0.4375) first step is 1.4375, second step is 1.4375 + 2*.4375 etc.
                cb.XTick = [1.4375,2.3135,3.1875,4.0625,4.9375,5.8125,6.6875,7.5625];
                
            end
            
            % Annotate the colorbar
            if plot_type_idx==1
                % Annotate chance for the decoding
                [~,chance_pos] = min(abs(linspace(cb_lim(1,1),cb_lim(1,2),num_timepoints)-((1/2)*100))); %1/2?
                ha = annotation('line','Color','k','LineWidth',1.5);
                ha.Parent = ax1;
                ha.X = [num_timepoints+25,num_timepoints+60];
                ha.Y = [chance_pos,chance_pos];
                ha = annotation('arrow','Color','k','LineWidth',1.5,'HeadStyle','vback1');
                ha.Parent = ax1;
                ha.X = [num_timepoints+25,num_timepoints+25];
                ha.Y = [chance_pos,chance_pos+40];
                text(num_timepoints+25,chance_pos+45,'Above chance','Rotation',90,'FontSize',font_size);
                text(num_timepoints+170,num_timepoints/2,'Decoding accuracy (%)','Rotation',90,'FontSize',font_size,'HorizontalAlignment','center');
            else
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
                
            end
            
            % set this before specifying font sizes otherwise they will be reset
            set(gca,'FontSize',font_size);
            
            % Set the axes
            ax = gca;
            set(ax,'Layer','top')
            if crossdecoding_direction == 1 %(touch->vis)
                ylabel('Training time (ms) - tactile trials', 'FontSize', font_size);
                xlabel('Testing time (ms) - visual trials', 'FontSize', font_size);
            elseif crossdecoding_direction == 2 %(vis->touch)
                ylabel('Training time (ms) - visual trials (ms)', 'FontSize', font_size);
                xlabel('Testing time (ms) - tactile trials (ms)', 'FontSize', font_size);
            end
            xlim([1,num_timepoints+1]);
            ylim([1,num_timepoints+1]);
            ax.YTick = t_idx;
            ax.YTickLabels = t_label;
            ax.XTick = t_idx;
            ax.XTickLabels = t_label;
            ax.XTickLabelRotation = 45;
            box on
            
            % Title
            if plot_type_idx==1
                text(-160,830,title_letter{invariance_idx},'FontSize',font_size+8,'HorizontalAlignment','left','FontWeight','bold'); % letter + sample
            end
            
            if crossdecoding_direction == 1 %(touch->vis)
                text(0,780,titles{plot_type_idx},'FontSize',font_size,'HorizontalAlignment','left','FontWeight','bold');
            elseif crossdecoding_direction == 2 %(vis->touch)
                text(0,780,titles{plot_type_idx + 2},'FontSize',font_size,'HorizontalAlignment','left','FontWeight','bold');
            end
            
        end
    end
    
    % Save plot
    fn = fullfile(current_dir,'plots',plot_fn{crossdecoding_direction});
    tn = tempname;
    print(gcf,'-dpng','-r500',tn)
    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=1;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
    
end
