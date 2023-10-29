% Script to run analyses reported in ICMPC 2023 paper
% Lauren Fink (finkl1@mcmaster.ca)
% Project started: MPIEA; currently: McMaster University 
% Last updated: 2023-05-28


% NOTE: This script provided as an example. It will not run, as the data are
% not included in the repository and the code relies on third-party
% toolboxes. 

%% Load and organize data
% Load in individual subject files

datadir = dir('/Users/lauren.fink/Documents/Projects/PupilLabs/memory/data/');
datadir = datadir(4:end); % Start at 4 to skip . and .. and .DS_Store

dt = table;
nr = 1;

% Loop through all directories and load
for k = 1:length(datadir)

    % define sub dir
    thissubdir = dir(fullfile(datadir(k).folder, datadir(k).name));
    thissubdir = thissubdir(4:end);
    thissubdir = thissubdir([thissubdir.isdir]); % get rid of meta files .json etc.

    % load all relevant files in sub directory
    for j = 1:length(thissubdir)

        % dir containing all files for this recording
        dt.sub{nr,1} = datadir(k).name;
        dt.folder{nr,1} = thissubdir(j).name; % note this has date/time and unique identifier
        thisrecordingdirpath = fullfile(thissubdir(j).folder, thissubdir(j).name);

        % give user feedback
        fprintf('Loading data from %s\n', thisrecordingdirpath);

        % load gaze data
        dt.gaze{nr,1} = readtable(fullfile(thisrecordingdirpath, 'gaze.csv'));

        % load events
        dt.events{nr,1} = readtable(fullfile(thisrecordingdirpath, 'events.csv'));
        numpiecestarts = sum(strcmp(dt.events{nr,1}.name, 'piece.begin'));
        nummistakes = sum(strcmp(dt.events{nr,1}.name, 'piece.mistake.stop'));
        nummistakes2 = sum(strcmp(dt.events{nr,1}.name, ' piece.mistake.stop'));
        dt.numperfs(nr,1) = numpiecestarts - (nummistakes+nummistakes2);

        % NOTE all below is fine but makes data frame huge. If not using
        % for now, don't load.
        % load audio from .mp4
        tempdir = dir(thisrecordingdirpath);
        audfile = dir(fullfile(thisrecordingdirpath, '*.mp4'));
        [dt.audsig{nr,1}, dt.audFs(nr,1)] = audioread(fullfile(audfile.folder, audfile.name));

        % load world timestamps
        %dt.worldts{nr,1} = readtable(fullfile(thisrecordingdirpath, 'world_timestamps.csv'));

        nr = nr+1;

    end
end

%% Create new table organized by performance

% sort dt by sub and folder so they are arranged in ascending order
% this will ensure we loop through in chronological order
dt = sortrows(dt,{'sub'; 'folder'});

% loop through all rows in data table and make performance table for each
% sub
for irow = 1:height(dt)
    performances = table();
    pr = 1;

    % parse performances depending on whether one or many per recording
    % (sometimes I did not stop the app between performances, but just let
    % it run throughout so as not to disturb the pianist). 

    % more than 1 performance
    if dt.numperfs(irow) > 1

        % find piece starts and ends
        piecestarts_mask = strcmp(dt.events{irow}.name, 'piece.begin');
        piecestart_inds = find(piecestarts_mask);

        % loop through each
        for ipiece = 1:numel(piecestart_inds)
            % grab events for just this piece
            if ipiece == numel(piecestart_inds)
                pieceevents = dt.events{irow}(piecestart_inds(ipiece):end, :);
            else
                pieceevents = dt.events{irow}(piecestart_inds(ipiece):piecestart_inds(ipiece+1)-1, :);
            end

            % check for mistakes. If mistake, don't analyse this piece
            mistake = strcmp(pieceevents.name, 'piece.mistake.stop') | strcmp(pieceevents.name, ' piece.mistake.stop');
            if sum(mistake) > 0
                continue
            end

            % grab and save piece name and perf num
            performances.sub{pr,1} = dt.sub{irow};
            performances.folder{pr,1} = dt.folder{irow};
            piecenum = pieceevents.name(contains(pieceevents.name, 'piece.name'));
            piecenum = str2double(piecenum{:}(end));
            performances.piece(pr,1) = piecenum;
            performances.perf(pr,1) = pr;

            % now we have only good data for one piece / performance
            sind = strcmp(pieceevents.name, 'piece.begin');
            stime = pieceevents.timestamp_ns_(sind);
            endind = strcmp(pieceevents.name, 'piece.end');
            etime = pieceevents.timestamp_ns_(endind);
            piecemask_gaze = dt.gaze{irow}.timestamp_ns_ >= stime & dt.gaze{irow}.timestamp_ns_ <= etime;
            performances.xdata{pr,1} = dt.gaze{irow}.gazeX_px_(piecemask_gaze);
            performances.ydata{pr,1} = dt.gaze{irow}.gazeY_px_(piecemask_gaze);
            ts_ns_thisperf = dt.gaze{irow}.timestamp_ns_(piecemask_gaze);
            performances.ts_secs_gaze{pr,1} =  (ts_ns_thisperf-ts_ns_thisperf(1)) / 1000000000; % convert to secs

            pr = pr+1;
        end

    % only one performance
    % NOTE will be same as most of what is above but just not in loop
    else
        % don't analyse this performance if mistake
        mistake = strcmp(dt.events{irow}.name, 'piece.mistake.stop') | strcmp(dt.events{irow}.name, ' piece.mistake.stop');
        if sum(mistake) > 0
            continue
        end

        performances.sub{pr,1} = dt.sub{irow};
        performances.folder{pr,1} = dt.folder{irow};

        % find piece starts and ends
        piecestart_mask = strcmp(dt.events{irow}.name, 'piece.begin');
        piecestart_ind = find(piecestart_mask);

        % grab and save piece name and perf num
        piecenum = dt.events{irow}.name(contains(dt.events{irow}.name, 'piece.name'));
        piecenum = str2double(piecenum{:}(end));
        performances.piece(pr,1) = piecenum;
        performances.perf(pr,1) = pr;

        % now we have only good data for one piece / performance
        sind = strcmp(dt.events{irow}.name, 'piece.begin');
        stime = dt.events{irow}.timestamp_ns_(sind);
        endind = strcmp(dt.events{irow}.name, 'piece.end');
        etime = dt.events{irow}.timestamp_ns_(endind);
        piecemask_gaze = dt.gaze{irow}.timestamp_ns_ >= stime & dt.gaze{irow}.timestamp_ns_ <= etime;
        performances.xdata{pr,1} = dt.gaze{irow}.gazeX_px_(piecemask_gaze);
        performances.ydata{pr,1} = dt.gaze{irow}.gazeY_px_(piecemask_gaze);
        ts_ns_thisperf = dt.gaze{irow}.timestamp_ns_(piecemask_gaze);
        performances.ts_secs_gaze{pr,1} =  (ts_ns_thisperf-ts_ns_thisperf(1)) / 1000000000; % convert to secs
    end

    % add performances table to larger data structure
    dt.performances{irow,1} = performances;
end

%% Concatenate all performance data for each sub. Cut to min len

pt = vertcat(dt.performances{:});
pt = sortrows(pt, {'sub', 'piece', 'folder'});

subs = unique(pt.sub);
for isub = 1:numel(subs)
    submask = strcmp(pt.sub, subs(isub));
    pieces = unique(pt.piece(submask));
    for ipiece = 1:numel(pieces)
        piecemask = pt.piece == pieces(ipiece);
        compmask = submask & piecemask;
        compinds = find(compmask);
        perfnums = 1:numel(compinds);
        pt.perfnum(compmask) = perfnums;
    end
end
pt = movevars(pt, 'perfnum', 'Before', 'xdata');



%% Trim data so that all match in length w/in subs
for isub = 1:numel(subs)
    submask = strcmp(pt.sub, subs(isub));
    subdata = pt(submask,:);

    % trim data to same length and save to new col in our table
    for irow = 1:height(subdata)
        minlen = cellfun(@size, subdata.xdata, 'UniformOutput', false);
        minlen = cell2mat(minlen);
        minlen = min(minlen(:,1));
        subdata.xdata_trimmed{irow} = subdata.xdata{irow}(1:minlen);
        subdata.ydata_trimmed{irow}  = subdata.ydata{irow}(1:minlen);
    end

    % add also to pt table
    pt.xdata_trimmed(submask) = subdata.xdata_trimmed;
    pt.ydata_trimmed(submask) = subdata.ydata_trimmed;
end

%% Trim data so that all match in length between subs
minlen = cellfun(@size, pt.xdata_trimmed, 'UniformOutput', false);
minlen = cell2mat(minlen);
minlen = min(minlen(:,1));

% trim data to same length and save to new col in our table
for irow = 1:height(pt)
    pt.xdata_trimmed_all{irow} = pt.xdata_trimmed{irow}(1:minlen);
    pt.ydata_trimmed_all{irow}  = pt.ydata_trimmed{irow}(1:minlen);
end

%% Notes
% Initially, I was thinking the analysis should be subject specific,
% allowing different minimum lengths between subjects, but trimming all to
% the same length allows for comparison across subjects to see, for example,
% if there is more within vs between subject similarity. Both versions of
% the trimmed data are valuable for different reasons. 


%% Organize data for similarity analysis

% get all data in matrix
xdata_all = cell2mat(pt.xdata_trimmed_all');
ydata_all = cell2mat(pt.ydata_trimmed_all');

% get meaningful names for rows
rownames_all = cell(height(pt),1);
for irow = 1:height(pt)
    rownames_all{irow,1} = strcat(pt.sub{irow}, '-', num2str(pt.piece(irow)), '-', num2str(pt.perfnum(irow)));
end

%% Just do simple corr mat for now to see if anything
% dtw much more computationally costly

distx = corrcoef(xdata_all);
disty = corrcoef(ydata_all);

figure()
imagesc(distx)
yticks = linspace(1, height(pt), numel(rownames_all));
set(gca, 'YTick', yticks, 'YTickLabel', rownames_all(:), 'TickLabelInterpreter','none')
set(gca, 'XTick', yticks, 'XTickLabel', rownames_all(:), 'TickLabelInterpreter','none')
colorbar()

figure()
imagesc(disty)


%% Calculate dtw matrix for each sub
sub1x = cell2mat(pt.xdata_trimmed(strcmp(pt.sub, 's01'))');
distx_1 = dtwdist(sub1x);
distx_1 = pdist(sub1x', @dtw);
figure()
imagesc(distx_1)
%disty = dtwdist(ydata_all);

%% 
s2 = size(sub1x, 2);
dtwmat = zeros(s2,s2);
for col1 = 1:s2
  for col2 = col1+1:s2
     Dist = dtw(sub1x(:, col1), sub1x(:, col2));
     dtwmat(col1,col2) = Dist;
  end
end

%% Loop through all subjects and calculate dtw

% create new table to save results to
nt = table;
nr = 1;

subs = unique(pt.sub);
for isub = 1:numel(subs)
    % Organize data
    currsub = subs(isub);
    subdatax = cell2mat(pt.xdata_trimmed(strcmp(pt.sub, currsub))');
    subdatay = cell2mat(pt.ydata_trimmed(strcmp(pt.sub, currsub))');
    s2 = size(subdatax, 2);
    dtwmatx = zeros(s2,s2);
    dtwmaty = zeros(s2,s2);
    
    % Calculate distances
    % x data
    sprintf('\nProcessing sub: %d, X data\n', isub)
    for col1 = 1:s2
        for col2 = col1+1:s2
            Distx = dtw(subdatax(:, col1), subdatax(:, col2));
            dtwmatx(col1,col2) = Distx;
        end
    end

    % y data
    sprintf('Processing sub: %d, Y data\n', isub)
    for col1 = 1:s2
        for col2 = col1+1:s2
            Disty = dtw(subdatay(:, col1), subdatay(:, col2));
            dtwmaty(col1,col2) = Disty;
        end
    end

    % save all to table
    sprintf('Saving data for sub: %d\n', isub)
    nt.sub{nr,1} = currsub;
    nt.labels{nr,1} = sublabels;
    nt.xdata{nr,1} = subdatax;
    nt.ydata{nr,1} = subdatay;
    nt.dtwx{nr,1} = dtwmatx;
    nt.dtwy{nr,1} = dtwmaty;

    nr = nr+1;
end

%% Calculate w/in vs between data for x and y

% Calculate w/in vs between piece similarity
for irow = 1:height(nt)
    sublabels = nt.labels{irow};
    currsub = subs{irow};
    piece1mask = contains(sublabels, strcat(currsub, '-1'));
    piece2mask = contains(sublabels, strcat(currsub, '-2'));
    
    % xin 
    indatax1 = nt.dtwx{irow}(piece1mask, piece1mask);
    indatax1 = indatax1(indatax1 >0);
    indatax2 = nt.dtwx{irow}(~piece1mask, ~piece1mask); %Todo
    indatax2 = indatax2(indatax2 >0);
    allin_x = vertcat(indatax1, indatax2);

%     % figure
%     figure()
%     subplot(3,1,1)
%     imagesc(nt.dtwx{irow})
% 
%     subplot(3,1,2)
%     test = nt.dtwx{irow};
%     test(piece1mask, piece1mask) = 10;
%     imagesc(test)
% 
%     subplot(3,1,3)
%     test = nt.dtwx{irow};
%     test(~piece1mask, ~piece1mask) = 10;
%     imagesc(test)
    
    %xout
    outdatax1 = nt.dtwx{irow}(piece1mask, piece2mask); 
    outdatax1 = outdatax1(outdatax1 >0);
%     outdatax2 = nt.dtwx{irow}(~piece2mask, ~piece2mask);
%     outdatax2 = outdatax2(outdatax2 >0);
    allout_x = outdatax1; %vertcat(outdatax1, outdatax2);

% %     % figure
%     figure()
%     subplot(3,1,1)
%     imagesc(nt.dtwx{irow})
% 
%     subplot(3,1,2)
%     test = nt.dtwx{irow};
%     test(piece1mask, piece2mask) = 10;
%     imagesc(test)
% 
%     subplot(3,1,3)
%     test = nt.dtwx{irow};
%     test(piece2mask, piece1mask) = 10;
%     imagesc(test)

    % save
    nt.x1_in{irow} = indatax1;
    nt.x2_in{irow} = indatax2;
    nt.x_all_in{irow} = allin_x;
    nt.x_in_mean(irow) = mean(allin_x);
    nt.x_in_sem(irow) = std(allin_x) / sqrt(numel(allin_x));

    nt.x1_out{irow} = outdatax1;
    nt.x2_out{irow} = outdatax2;
    nt.x_all_out{irow} = allout_x;
    nt.x_out_mean(irow) = mean(allout_x);
    nt.x_out_sem(irow) = std(allout_x) / sqrt(numel(allout_x));

    
    % now same for y data
    % yin
    indatay1 = nt.dtwy{irow}(piece1mask, piece1mask);
    indatay1 = indatay1(indatay1 >0);
    indatay2 = nt.dtwy{irow}(~piece1mask, ~piece1mask);
    indatay2 = indatay2(indatay2 >0);
    allin_y = vertcat(indatay1, indatay2);
    
    % yout
    outdatay1 = nt.dtwy{irow}(piece1mask, piece2mask);
    outdatay1 = outdatay1(outdatay1 >0);
%     outdatay2 = nt.dtwy{irow}(piece2mask, piece1mask);
%     outdatay2 = outdatay2(outdatay2 >0);
    allout_y = outdatay1; %vertcat(outdatax1, outdatay2);

    % save
    nt.y1_in{irow} = indatay1;
    nt.y2_in{irow} = indatay2;
    nt.y_all_in{irow} = allin_y;
    nt.y_in_mean(irow) = mean(allin_y);
    nt.y_in_sem(irow) = std(allin_y) / sqrt(numel(allin_y));
    
    nt.y1_out{irow} = outdatay1;
    nt.y2_out{irow} = outdatay2;
    nt.y_all_out{irow} = allout_y;
    nt.y_out_mean(irow) = mean(allout_y);
    nt.y_out_sem(irow) = std(allout_y) / sqrt(numel(allout_y));


end
%% Calculate ttest for within vs. between piece similarity

% xdata
in_allsubs_x = vertcat(cell2mat(nt.x_all_in));
out_allsubs_x = vertcat(cell2mat(nt.x_all_out));
[h, p, ci, stats] = ttest2(in_allsubs_x, out_allsubs_x)

% ydata
in_allsubs_y = vertcat(cell2mat(nt.y_all_in));
out_allsubs_y = vertcat(cell2mat(nt.y_all_out));
[h, p, ci, stats] = ttest2(in_allsubs_y, out_allsubs_y)

%% Plot w/in vs between mean and sem (x data)
figure()
model_series = [mean(nt.x_in_mean); mean(nt.x_out_mean)]; % in and out means
model_error = [mean(nt.x_in_sem); mean(nt.x_out_sem)];
b = bar(model_series, 'grouped')
b.FaceColor = 'flat';
b.CData(1,:) = [0.9290 0.6940 0.1250];
b.CData(2,:) = [0.6350 0.0780 0.1840];

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);
% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
% Plot the errorbars
errorbar(x',model_series,model_error, 'Color', 'k','linestyle','none');
hold off

set(gca,'XTickLabel',{'Within piece','Between pieces'});
title('Avg. similarity w/in vs. between pieces')
ylabel('Eucl. Dist. [a.u.]')
set(gca, 'box', 'off')
set(gca, 'Fontsize', 16)
set(gca, 'LineWidth', 2)


%% Plot w/in vs between mean and sem (y data)
figure()
model_series = [mean(nt.y_in_mean); mean(nt.y_out_mean)]; % in and out means
model_error = [mean(nt.y_in_sem); mean(nt.y_out_sem)];
b = bar(model_series, 'grouped')
b.FaceColor = 'flat';
b.CData(1,:) = [0.9290 0.6940 0.1250];
b.CData(2,:) = [0.6350 0.0780 0.1840];

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);
% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i,:) = b(i).XEndPoints;
end
% Plot the errorbars
errorbar(x',model_series,model_error, 'Color', 'k','linestyle','none');
hold off

set(gca,'XTickLabel',{'Within piece','Between pieces'});
%title('Avg. similarity w/in vs. between pieces (y data)')
ylabel('Eucl. Dist. [a.u.]')
set(gca, 'box', 'off')
set(gca, 'Fontsize', 16)
set(gca, 'LineWidth', 2)




%% Plot traces as an example
figure()
stackedplot(nt.xdata{1,1}, 'k');
set(gca, 'Fontsize', 16)
set(gca, 'LineWidth', 2)
%%
figure()
stackedplot(nt.xdata{2,1}, 'k');
set(gca, 'Fontsize', 16)
set(gca, 'LineWidth', 2)
axis off
%%
figure()
stackedplot(nt.xdata{3,1}, 'k');
set(gca, 'Fontsize', 16)
set(gca, 'LineWidth', 2)


%% mess with mp and skimp
data = nt.xdata{2};
figure()
for i = 1:size(data,2)
    [matrixProfile{i}, profileIndex{i}, motifIdxs{i}, discordIdx{i}] = ...
    interactiveMatrixProfileVer3(data(:,i), 1000);
end


%%
figure()
for i = 1:size(data,2)
    subplot(2,12,i)
    imagesc(PMP{i})
    
    % multiple y axis by seq. len
    ylabs = get(gca, 'YTickLabel');
    for ilab = 1:numel(ylabs)
        ylabs{ilab} = str2double(ylabs{ilab})*100;
    end
    set(gca,'YTickLabel',flip(ylabs)); 
end

%% Plot avg profile
figure()
% avgpmp = PMP(:,1:12)';
% avgpmp = cellfun(@mean, avgpmp);

% extend 1 out longer
longd = vertcat(data(:,1), data(:,1));
[PMP, ~] = SKIMP(longd, 1:100:numel(longd)/2, 'ShowImage', false);
figure()
imagesc(PMP)
ylabs = get(gca, 'YTickLabel');
ylabs{:} = str2double(ylabs{:})*100;
set(gca,'YTickLabel',flip(ylabs));
colorbar()


meanPMP = mean(cat(3, avgpmp{:}), 3);
figure()
imagesc(meanPMP)
axis ij
axis square
shading interp
% multiple y axis by seq. len
ylabs = get(gca, 'YTickLabel');
for ilab = 1:numel(ylabs)
    ylabs{ilab} = str2double(ylabs{ilab})*100;
end
set(gca,'YTickLabel',flip(ylabs));
colorbar()

figure()
pcolor(meanPMP)
axis ij
axis square
shading interp
colorbar()
ylabs = get(gca, 'YTickLabel');
ylabs = 10:100:2000;
set(gca,'YTickLabel',flip(ylabs));



%% 
data = nt.xdata{2}';
data = data(1,:);
[best_so_far,motiffirst,motifsec] = DTWMotifDiscoveryGUI(data,1000,5)



%% Compare audio sig motifs to eye ones
% just read in one audio file for now
istim = 37;
audsig = dt.audsig{istim}; % corresponds to sub 2, rec 1

% figure()
% plot(audsig)

% cut audio sig
% from visual inspection of waveform
cleansig = audsig(1350080:3956620);

figure()
plot(cleansig)

test = cleansig(1:100000);

audFs = dt.audFs(istim);
N = audFs *.005; % 50ms - 20 hz. 
[YUPPER,YLOWER] = envelope(cleansig, N, 'rms');
[p,q] = rat(200 / audFs); 
yu = resample(YUPPER, p, q);

figure()
plot(yu)



%% re-start piece mus mp comparison
% THIS
ed = pt.xdata{37};

[matrixProfile_e, profileIndex_e, motifIdxs_e, discordIdx_e] = ...
    interactiveMatrixProfileVer3(ed, 2100)

[PMP_ed, IDX_ed] = SKIMP(ed, 1:200:numel(ed)/2, 'ShowImage', true);

%% 
figure()
imagesc(PMP_ed)
title('PMP eye data')
% multiple y axis by seq. len
ylabs = get(gca, 'YTickLabel');
for ilab = 1:numel(ylabs)
    ylabs{ilab} = str2double(ylabs{ilab})*200;
end
set(gca,'YTickLabel',flip(ylabs));
colorbar()

%% now audio
% cut audio sig
% from visual inspection of waveform
cleansig = audsig(1350080:3945790);
N = audFs*.005; 
[YUPPER,YLOWER] = envelope(cleansig, N, 'rms');
[p,q] = rat(200 / audFs); 
yu = resample(YUPPER, p, q);
figure()
plot(yu)

yu = yu(1:size(ed,1)); % now mus and audio same (were already very close
%% 
[matrixProfile, profileIndex, motifIdxs, discordIdx] = ...
    interactiveMatrixProfileVer3(yu, 1500) %2000

[PMP_yu, IDX_yu] = SKIMP(yu, 1:200:numel(yu)/2, 'ShowImage', true);

%%
figure()
imagesc(PMP_yu)
title('PMP music data')
% multiple y axis by seq. len
ylabs = get(gca, 'YTickLabel');
for ilab = 1:numel(ylabs)
    ylabs{ilab} = str2double(ylabs{ilab})*200;
end
set(gca,'YTickLabel',flip(ylabs));
colorbar()

%% Play sound at moments of two eye motifs (1500)
% need to convert from 200 hz samples to 44.8k
% 3749 and 5367 (+1500)

constoffset = 1350080; % use original file import which is still double
auds = audFs*3749 / 200;
auds2 = audFs*5367 / 200;
audwin = audFs*1500 / 200;

p = audioplayer(audsig(constoffset+auds:constoffset+auds+audwin), audFs);
play(p)
audiowrite('motif1.wav',p,audFs)

p2 = audioplayer(audsig(constoffset+auds2:constoffset+auds2+audwin), audFs);
play(p2)


soundsc(audsig(constoffset+auds:constoffset+auds+audwin))


%% Play sound at moments of two eye motifs (2000)
% need to convert from 200 hz samples to 44.8k
% 27 and 8240 (+2000)

constoffset = 1350080; % use original file import which is still double
auds = audFs*13 / 200;
auds2 = audFs*8226 / 200;
audwin = audFs*2100 / 200;

p = audioplayer(audsig(constoffset+auds:constoffset+auds+audwin), audFs);
play(p)
audiowrite('motif1_2100.wav',audsig(constoffset+auds:constoffset+auds+audwin),audFs)

p2 = audioplayer(audsig(constoffset+auds2:constoffset+auds2+audwin), audFs);
play(p2)
audiowrite('motif2_2100.wav',audsig(constoffset+auds2:constoffset+auds2+audwin),audFs)


%% Loaded in all audio had saved previously
% figure()
% plot(motif1)
% hold on 
% plot(motif2, 'r')

% amp env
[upper_m1, lower] = envelope(motif1, 2400, 'rms');
[b,a] = butter(3, 50/(fs/2), 'low'); % filter
filtenv = filtfilt(b,a,upper_m1(:,1)); % filter amp env?
normenv = zscore(filtenv);

[upper_m2, lower] = envelope(motif2, 2400, 'rms');
[b,a] = butter(3, 50/(fs/2), 'low'); % filter
filtenv = filtfilt(b,a,upper_m2(:,1)); % filter amp env?
normenv = zscore(filtenv);

figure()
plot(upper_m1)
hold on 
plot(upper_m2)

% get pitch of each and plot
 % pitch
 [f0_m1,idx_m1] = pitch(motif1,fs, ...
     'Method','PEF', ...
     'Range', [20, 3300], ...
     'WindowLength',4800, ... % 2400 = 50 ms window. 10 ms wins
     'OverlapLength',3600, ... % 1800 = 75% overlap
     'MedianFilterLength',1); % 1 = no smoothing

 [f0_m2,idx_m2] = pitch(motif2,fs, ...
     'Method','PEF', ...
     'Range', [20, 3300], ...
     'WindowLength',4800, ... % 50 ms window. 10 ms wins
     'OverlapLength',3600, ... % 75% overlap
     'MedianFilterLength',1); % 1 = no smoothing


figure()
plot(f0_m1)
hold on 
plot(f0_m2)


%%
% amp env
[upper_m1, lower] = envelope(motif1_21, 2400, 'rms');
[b,a] = butter(3, 50/(fs/2), 'low'); % filter
filtenv = filtfilt(b,a,upper_m1(:,1)); % filter amp env?
normenv = zscore(filtenv);

[upper_m2, lower] = envelope(motif2_21, 2400, 'rms');
[b,a] = butter(3, 50/(fs/2), 'low'); % filter
filtenv = filtfilt(b,a,upper_m2(:,1)); % filter amp env?
normenv = zscore(filtenv);

figure()
plot(upper_m1)
hold on 
plot(upper_m2)

% get pitch of each and plot
 % pitch
 [f0_m1,idx_m1] = pitch(motif1_21,fs, ...
     'Method','PEF', ...
     'Range', [20, 3300], ...
     'WindowLength',4800, ... % 2400 = 50 ms window. 10 ms wins
     'OverlapLength',3600, ... % 1800 = 75% overlap
     'MedianFilterLength',1); % 1 = no smoothing

 [f0_m2,idx_m2] = pitch(motif2_21,fs, ...
     'Method','PEF', ...
     'Range', [20, 3300], ...
     'WindowLength',4800, ... % 50 ms window. 10 ms wins
     'OverlapLength',3600, ... % 75% overlap
     'MedianFilterLength',1); % 1 = no smoothing


figure()
plot(f0_m1)
hold on 
plot(f0_m2)