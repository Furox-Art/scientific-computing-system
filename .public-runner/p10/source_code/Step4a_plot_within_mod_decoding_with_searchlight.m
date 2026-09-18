%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 4a - plot within-modality decoding including searchlight
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: cbrewer, matplotlib, fieldtrip

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_bids';
ft_defaults

%% Set the participant number
VT_group = [8	9	12	13	14	16	18	20	21	23	28	29	30	31	32	33	35	36	38	39	40];
NoVT_group = [1	2	3	4	5	6	7	10	11  15	17	19	22	24  25	26	27	34	37];

% Pre-allocate variables
timepoints = 744;
mean_dec_vis = zeros(2,timepoints);
mean_dec_touch = zeros(2,timepoints);
CI_vis_finger = zeros(2,timepoints,2);
CI_touch_finger = zeros(2,timepoints,2);
bf_touch_finger = zeros(timepoints,2);
bf_vis_finger = zeros(timepoints,2);
dec_acc_vis_over_groups = cell(1,2);
dec_acc_touch_over_groups = cell(1,2);
res_vision_sl = cell(1,2);
res_touch_sl = cell(1,2);

for samples = 1:2 %(VT and noVT)
    
    if samples == 1
        subs_all = VT_group;
    else
        subs_all = NoVT_group;
    end
    
    subs_excluded = [10,13,16,18,32,33]; %excluded based on task performance <60%
    sub_ids = setdiff(subs_all,subs_excluded);
    num_subs = size(sub_ids,2);
    
    %% Load the data
    % loop over participats
    res_vision = [];
    res_touch = [];
    
    for sub_idx = 1:num_subs
        
        %Get the participant ID
        sub_num = sub_ids(1,sub_idx);
        fprintf('Loading data S%.0f of %.0f\n',sub_idx,num_subs);
        load(fullfile(current_dir,'derivatives','decoding',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_decoding.mat']));
        
        %Create empty arrays for decoding only the first loop so it doesn't get wiped
        if sub_idx == 1
            dec_acc_vis_finger      = zeros(size(sub_ids,2),timepoints);
            dec_acc_touch_finger    = zeros(size(sub_ids,2),timepoints);
        end
        
        % Add the decoding accuracy
        dec_acc_vis_finger(sub_idx,:)    = dec_acc_som(1,:)*100;
        dec_acc_touch_finger(sub_idx,:)  = dec_acc_som(2,:)*100;
        
        % Add data for this ppt
        res_vision{sub_idx} = res_searchlight{1};
        res_touch{sub_idx} = res_searchlight{2};
        
        
    end
    
    % Calculate the mean decoding
    mean_dec_vis(samples,:) = mean(dec_acc_vis_finger);
    mean_dec_touch(samples,:) = mean(dec_acc_touch_finger);
    
    % Save the data per group so we can compare them
    dec_acc_vis_over_groups{samples} = dec_acc_vis_finger;
    dec_acc_touch_over_groups{samples} = dec_acc_touch_finger;
    
    % Get the searchlight data
    res_vision_sl{samples} = cosmo_stack(res_vision);
    res_touch_sl{samples} = cosmo_stack(res_touch);
    
    %% Calculate the CI
    n_boot = 10000;
    rng(1);
    CI_vis_finger(:,:,samples) = bootci(n_boot,{@mean,dec_acc_vis_finger},'type','per');
    CI_touch_finger(:,:,samples) = bootci(n_boot,{@mean,dec_acc_touch_finger},'type','per');
    
    %% Calculate the BF
    bf_vis_finger(:,samples) = bayesfactor_R_wrapper(dec_acc_vis_finger','args','mu=50,rscale="medium",nullInterval=c(0.5,Inf)');
    bf_touch_finger(:,samples) = bayesfactor_R_wrapper(dec_acc_touch_finger','args','mu=50,rscale="medium",nullInterval=c(0.5,Inf)');
    
end

%% Calculate the BF for the difference between groups
bf_vis_finger(:,3) = bayesfactor_R_wrapper_2sample(dec_acc_vis_over_groups{1}',dec_acc_vis_over_groups{2}','returnindex',2,'args','mu=0,rscale="medium",nullInterval=c(-0.5,0.5)');
bf_touch_finger(:,3) = bayesfactor_R_wrapper_2sample(dec_acc_touch_over_groups{1}',dec_acc_touch_over_groups{2}','returnindex',2,'args','mu=0,rscale="medium",nullInterval=c(-0.5,0.5)');

%% Figure specifications
p_col = tab10; % you need matplotlib for this
p_col = p_col([1,2,4],:);
font_size = 16;
font_size_legend = 13;
titles = {'Visual: finger/thumb decoding' 'Tactile: finger/thumb decoding'};
title_letter = [{'b'},{'a'}];
bf_labels = [{'Vicarious touch group'}, {'No vicarious touch group'},{'Difference'}];
plot_fn = 'within_mod_searchlight';

% specify time windows
time = ds.a.fdim.values{2};
timewins = [(-1000:250:1500)' (-750:250:1750)']; % skip baseline
num_timepoints = 744;
t_idx = -1000:250:1750;

% specify plotting range
max_sl_vis = max(mean([res_vision_sl{1}.samples;res_vision_sl{2}.samples]-0.5));
max_sl_touch = max(mean([res_touch_sl{1}.samples;res_touch_sl{2}.samples]-0.5));
mrange = [0, round(max_sl_vis*100)/100; 0, round(max_sl_touch*100)/100];
bf_lim = [-4,4;-6,6];

% combine
Mod = cat(3, mean_dec_vis, mean_dec_touch);
CIs = cat(4, CI_vis_finger, CI_touch_finger);
BFs = cat(3, bf_vis_finger, bf_touch_finger);

%% plot
f = figure(1); clf;
f.Position=[f.Position(1:2) 1000 1200];
f.PaperPositionMode='auto';
f.Resize='off';

y_pos = [0.28,0.78];
y_pos_bf = [0.24,0.205,0.17;0.74,0.705,0.67];
y_pos_sl = [0.055,0.0005;0.555,0.5005];

for modality = 1:2 % 1 = vision, 2 = touch
    
    CI_dec = CIs(:,:,:,modality);
    BF_dec = BFs(:,:,modality);
    Mod_dec = Mod(:,:,modality);
    title_dec = titles{modality};
    
    %%% Letter %%%
    ax_title = axes('Position',[0.02,y_pos(modality)+0.2+0.015,0.1,0.1],'LineWidth',1.5);
    th = text(0,0,title_letter{modality},'FontSize',font_size+6,'FontWeight','bold');
    ax_title.Visible = 'off';
    
    %%% PLOT DECODING %%%
    ax1 = axes('Position',[0.1,y_pos(modality),0.76,0.2],'LineWidth',1.5);
    hold on
    line([time(1),time(end)],[50,50],'color','k','LineWidth',1.5);
    line([0,0],[48,56],'color','k','LineWidth',1.5);
    text(-50,55.5,'Touch onset','FontSize',font_size,'HorizontalAlignment','right','Rotation',90);
    if modality==1
        line([-1000,-1000],[0,100],'color','k','LineWidth',1.5);
        text(-950,55.5,'Movie onset','FontSize',font_size,'HorizontalAlignment','right','Rotation',90);
    end
    fill([time,fliplr(time)],[CI_dec(1,:,1),fliplr(CI_dec(2,:,1))],p_col(1,:),'FaceAlpha',0.2,'LineStyle','none')
    fill([time,fliplr(time)],[CI_dec(1,:,2),fliplr(CI_dec(2,:,2))],p_col(2,:),'FaceAlpha',0.2,'LineStyle','none')
    ph1 = plot(time,Mod_dec(1,:),'LineWidth',1,'color',p_col(1,:),'LineWidth',1.5);
    ph2 = plot(time,Mod_dec(2,:),'LineWidth',1,'color',p_col(2,:),'LineWidth',1.5);
    xlim([time(1),time(end)]);
    if modality == 1
        ylim([48,56]);
    else
        ylim([48,56]);
    end
    legend([ph1,ph2],[{'Vicarious touch group'},{'No vicarious touch group'}]);
    legend boxoff;
    
    % set this before specifying font sizes otherwise they will be reset
    set(gca,'FontSize',font_size);
    
    ax1.XTick = t_idx;
    ax1.XTickLabels = [];
    ylabel('Decoding accuracy (%)', 'FontSize', font_size+4);
    title(title_dec, 'FontSize', font_size+4);
    
    %Plot BFs
    for group = 1:3
        ax = axes('Position',[0.1,y_pos_bf(modality,group),0.76,0.024],'LineWidth',1.5);
        hold on
        line([0,0],bf_lim(modality,:),'Color','k','LineWidth',1.5);
        if modality==1
            line([-1000,-1000],bf_lim(modality,:),'color','k','LineWidth',1.5);
        end
        
        % Make the colormap
        num_cols = 200;
        cmap = twilight(num_cols);
        cmap = [flipud(cmap(1:num_cols/2,:));flipud(cmap(num_cols/2+1:end,:))];
        val_col_map = logspace(bf_lim(modality,1),bf_lim(modality,2),num_cols);
        
        % Plot the BFs
        line([time(1),time(end)],[0,0],'color','k','LineWidth',1.5);
        for i = 1:size(BF_dec(:,group),1)
            [~,idx] = min(abs(val_col_map-BF_dec(i,group)));
            scatter(time(i),log10(BF_dec(i,group)),[],'k','filled','SizeData',16);
            scatter(time(i),log10(BF_dec(i,group)),[],cmap(idx,:),'filled','SizeData',14);
        end
        text(max(time)-0.5,bf_lim(modality,2)+diff(bf_lim(modality,:))*0.1,bf_labels{group},'FontWeight','bold','FontSize',font_size,'HorizontalAlignment','right');
        ax.YLim = bf_lim(modality,:);
        ax.YTick = [bf_lim(modality,1),bf_lim(modality,2)];
        ax.YTickLabel = [{['10^{',num2str(bf_lim(modality,1)),'}']},{['10^{',num2str(bf_lim(modality,2)),'}']}];
        xlim([time(1),time(end)]);
        if group==2
            yl = ylabel({'BF';'(Log scale)'});
        end
        set(gca,'FontSize',font_size);
        
        if group==3
            xlabel('Time (ms)');
            ax.XTick = t_idx;
            ax.XTickLabels = t_idx;
        else
            ax.XTick = t_idx;
            ax.XTickLabels = [];
        end
        if group==2
            text(1870,1,'BF','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center');
            text(1920,1,'(Log scale)','FontSize',font_size,'Rotation',90,'HorizontalAlignment','center');
        end
    end
    
    % Colorbar
    colormap(ax,cmap);
    cb = colorbar;
    cb.Limits = [0,1];
    cb.Position = [0.91,y_pos_bf(modality,3),0.015,0.09];
    cb.LineWidth = 1;
    cb.Ticks = [0,0.5,1];
    cb.TickLabels = [{['10^{',num2str(bf_lim(modality,1)),'}']},{'1'},{['10^{',num2str(bf_lim(modality,2)),'}']}];
    cb.FontSize = font_size;
    
    % Plot the sensor searchlight
    for group = 1:2
        % Get the searchlight data
        if modality==1
            res = res_vision_sl{group};
        else
            res = res_touch_sl{group};
        end
        res.samples = res.samples-0.5;
        ft = ft_timelockanalysis([],cosmo_map2meeg(res));
        
        % Find the electrode layout
        % NOTE: biosemi has label 'Afz', but fieldtrip has label 'AFz'
        layout = cosmo_meeg_find_layout(res,'label_threshold',.99);
        % Select the electrodes we use
        idx = ismember(layout.label,res.a.fdim.values{1});
        layout.pos = layout.pos(idx,:);
        layout.width = layout.width(idx,:);
        layout.height = layout.height(idx,:);
        layout.label = layout.label(idx,:);
        
        % Loop over time windows
        for ttt = 1:length(timewins)
            
            % Set the position for this figure
            bfh = .1;
            aw = (0.745-(0.745*(100/(timewins(end)-timewins(1)))))./length(timewins);
            a = axes('Position',[0.095+ttt*aw-(aw/2),y_pos_sl(modality,group)-0.004,aw,0.98*bfh]);hold on
            % show figure with plots for each sensor
            cfg = [];
            cfg.zlim = mrange(modality,:);
            cfg.xlim = timewins(ttt,:);
            cfg.layout = layout;
            cfg.showscale = 'no';
            cfg.comment = 'no';
            cfg.markersymbol = '.';
            cfg.markersize = 6;
            cfg.figure = 'gca';
            cfg.style = 'straight';
            cfg.gridscale = 100;
            ph = ft_topoplotER(cfg, ft);
            a.FontSize = font_size_legend;
            a.Colormap = [linspace(1,p_col(group,1),10);linspace(1,p_col(group,2),10);linspace(1,p_col(group,3),10)]';
            set(a.Children,'LineWidth',1);
        end
        
        % Add the colorbar
        ax = axes('Position',[0.88,y_pos_sl(modality,group)+0.045,0.075,0.05]);
        ax.Colormap = [linspace(1,p_col(group,1),10);linspace(1,p_col(group,2),10);linspace(1,p_col(group,3),10)]';
        ax.Visible = 'off';
        cb = colorbar('south');
        cb.Ticks = [];
        text(0,-0.08,num2str((mrange(modality,1)+0.5)*100),'HorizontalAlignment','left','FontSize',font_size_legend);
        text(1,-0.08,num2str((mrange(modality,2)+0.5)*100),'HorizontalAlignment','right','FontSize',font_size_legend);
        text(0.5,0.35,'Decoding acc. (%)','HorizontalAlignment','center','FontSize',font_size_legend);
    end
    
    
end

%% Save plot
fn = fullfile(current_dir,'plots',plot_fn);
tn = tempname;
print(gcf,'-dpng','-r1000',tn)
im=imread([tn '.png']);
[i,j]=find(mean(im,3)<255);margin=2;
imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
