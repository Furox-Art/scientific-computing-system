%--------------------------------------------------------------------------
% EEG decoding - Vicarious touch and No vicarious touch groups
% Step 4d - plot between-modality decoding searchlight
%--------------------------------------------------------------------------
% The following toolboxes need to be on the path: cbrewer, matplotlib, fieldtrip

clearvars; clc;

%% Set the path
current_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_test';
ft_defaults

%% Set the participant number
VT_group = [8	9	12	13	14	16	18	20	21	23	28	29	30	31	32	33	35	36	38	39	40];
NoVT_group = [1	2	3	4	5	6	7	10	11  15	17	19	22	24  25	26	27	34	37];

res_crossdecoding_sl_stack = cell(1,2);

for samples = 1:2 %(VT & no VT)
    
    if samples == 1
        subs_all = VT_group;
    else
        subs_all = noVT_group;
    end
    
    subs_excluded = [10,13,16,18,32,33]; %excluded based on task performance <60%
    sub_ids = setdiff(subs_all,subs_excluded);
    num_subs = size(sub_ids,2);
    
    %% Load the data
    % loop over participats
    res_crossdecoding_sl = cell(1,num_subs);
    
    for sub_idx = 1:num_subs
        
        %Get the participant ID
        sub_num = sub_ids(1,sub_idx);
        fprintf('Loading data S%.0f of %.0f\n',sub_idx,num_subs);
        
        % Load sensor seachlight data
        load(fullfile(current_dir,'derivatives','crossdecoding_searchlight',[num2str(sub_num,'sub-%02d'),'_task-touchdecoding_crossdecoding_SL.mat']));
        
        % Add data for this ppt
        res_crossdecoding_sl{sub_idx} = cdt_tf_ds;
        
    end
    
    % Stack the searchlight data
    res_crossdecoding_sl_stack{samples} = cosmo_stack(res_crossdecoding_sl);
    
end

%% Figure specifications
p_col = tab10; % you need matplotlib for this
p_col = p_col([2,1],:);
font_size = 26;
font_size_legend = 13;

% get the timescale labels
time = cdt_tf_ds.a.fdim.values{2};
timewins = [(-1000:250:1500)' (-750:250:1750)'];
twin = [1,4,5,11];
plot_fn = [{'between_mod_searchlight_VT'}, {'between_mod_searchlight_noVT'}];
titles = [{'a Vicarious touch group'},{'b No vicarious touch group'}];

num_timepoints = 744;
tv = cdt_tf_ds.a.fdim.values{2};

t_label = -1000:250:1750;
t_idx = zeros(1,12);
for i = 1:12
    [~,t_idx(1,i)] = min(abs(tv-t_label(i)));
end

% define range
mrange = [0, 0.02];

% location of the plots in the figure
y_pos_sl = linspace(0.1,0.79,size(timewins,1))+0.03;

%% plot
% Plot the sensor searchlight - show figure with plots for each time window
% train_time is on the vertical ('freq') axis
% test_time on the horizontal ('time') axis)

for samples = 1
    
    f = figure(samples); clf;
    f.Position=[f.Position(1:2) 1000 1000];
    f.PaperPositionMode='auto';
    f.Resize='off';
    
    % Get the searchlight data
    res = res_crossdecoding_sl_stack{samples};
    res.samples = res.samples-0.5;
    ft=cosmo_map2meeg(res);
    
    % Find the electrode layout
    % NOTE: biosemi has label 'Afz', but fieldtrip has label 'AFz'
    layout = cosmo_meeg_find_layout(res,'label_threshold',.99);
    % Select the electrodes we use
    idx = ismember(layout.label,res.a.fdim.values{1});
    layout.pos = layout.pos(idx,:);
    layout.width = layout.width(idx,:);
    layout.height = layout.height(idx,:);
    layout.label = layout.label(idx,:);
    
    % Make the big axes so we can have time labels
    ax = axes('Position',[0.115,0.118,0.782,0.797],'LineWidth',1.5);hold on
    box on;
    
    % Title
    text(-100,800,titles{samples},'FontSize',font_size,'HorizontalAlignment','left','FontWeight','bold');
    
    % Loop over training-time windows
    for train_time = 1:length(timewins)
        
        for test_time = 1:length(timewins)
            % Set the position for this figure
            bfh = .1;
            aw = (0.76-(0.76*(100/(time(end)-time(1)))))./length(timewins);
            a = axes('Position',[0.11+test_time*aw-(aw/2)+(0.0004*test_time)-0.0004,y_pos_sl(train_time),aw,0.98*bfh]);hold on
            cfg = [];
            cfg.zlim = mrange;
            cfg.ylim = timewins(train_time,:);
            cfg.xlim = timewins(test_time,:);
            cfg.layout = layout;
            cfg.showscale = 'no';
            cfg.comment = 'no';
            cfg.markersymbol = '.';
            cfg.markersize = 6;
            cfg.figure = 'gca';
            cfg.style = 'straight';
            cfg.gridscale = 100;
            ft_topoplotTFR(cfg,ft);
            a.FontSize = font_size_legend;
            a.Colormap = [linspace(1,p_col(samples,1),100);linspace(1,p_col(samples,2),100);linspace(1,p_col(samples,3),100)]';
            set(a.Children,'LineWidth',1.2);
            
        end
    end
    
    % Set the axes
    set(f,'CurrentAxes',ax);
    % Grid line at 0ms
    line([find(tv==0),find(tv==0)],[1,num_timepoints+1],'LineWidth',1.5,'Color','k');
    line([find(tv==-1000),find(tv==-1000)],[1,num_timepoints+1],'LineWidth',1.5,'Color','k');
    text(27,770,'Movie onset','FontSize',font_size,'HorizontalAlignment','center');
    text(283,770,'Touch onset','FontSize',font_size,'HorizontalAlignment','center');
    line([1,num_timepoints+1],[find(tv==0),find(tv==0)],'LineWidth',1.5,'Color','k');
    line([1,num_timepoints+1],[find(tv==-1000),find(tv==-1000)],'LineWidth',1.5,'Color','k');
    
    ylabel('Training time (ms) - tactile trials', 'FontSize', font_size);
    xlabel('Testing time (ms) - visual trials', 'FontSize', font_size);
    
    xlim([1,num_timepoints+1]);
    ylim([1,num_timepoints+1]);
    ax.YTick = t_idx;
    ax.YTickLabels = t_label;
    ax.XTick = t_idx;
    ax.XTickLabels = t_label;
    ax.XTickLabelRotation = 45;
    ax.FontSize = font_size;
    
    % Add the colorbar
    ax = axes('Position',[0.9,0.2,0.1,0.2]);
    ax.Colormap = [linspace(1,p_col(samples,1),100);linspace(1,p_col(samples,2),100);linspace(1,p_col(samples,3),100)]';
    ax.Visible = 'off';
    cb = colorbar;
    cb.Ticks = [0,2];
    cb.TickLabels = 50:52;
    cb.Position  = [0.943,0.118,0.02,0.797];
    cb.FontSize = font_size;
    text(0.2,1.58,'Decoding accuracy (%)','HorizontalAlignment','center','FontSize',font_size,'Rotation',90);
    
    
    %% Save plot
    fn = fullfile(current_dir,'plots',plot_fn{samples});
    tn = tempname;
    print(gcf,'-dpng','-r1000',tn)
    im=imread([tn '.png']);
    [i,j]=find(mean(im,3)<255);margin=2;
    imwrite(im(min(i-margin):max(i+margin),min(j-margin):max(j+margin),:),[fn '.png'],'png');
    
end
