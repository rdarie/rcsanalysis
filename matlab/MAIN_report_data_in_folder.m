function MAIN_report_data_in_folder(varargin)
if isempty(varargin)
    [dirname] = uigetdir(pwd,'choose a dir with rcs session folders');
else
    dirname  = varargin{1};
end
tblout = getDataBaseRCSdata(dirname);
% print out details about this to a text file in this directory 
fid = fopen(fullfile(dirname,'recordingReport.txt'),'w+');

for t = 1:size(tblout)
    if ~isempty(tblout.startTime{t})
        fprintf(fid,'%s (Duration)\n\t\t%s\t\t%s\n',...
            tblout.duration{t},tblout.startTime{t},tblout.endTime{t});
        et = tblout.eventData{t};
        if ~isempty(et)
            idxuse = ~cellfun(@(x) any(strfind(x, 'BatteryLevel')),et.EventType);
            if sum(idxuse) >=1
                etuse = et(idxuse,:);
                for e = 1:size(etuse,1)
                    fprintf(fid,'\t\t\t %s\n \t\t\t%s\n',...
                        etuse.EventSubType{e},etuse.EventType{e});
                end
                
            end
        end
    end
end

toSum = [tblout.duration{~cellfun(@(x) isempty(x),tblout.duration)}];
totalDuration = sum(toSum(toSum > duration(seconds(0))));
fnmsave = fullfile(dirname,'database.mat');
save(fnmsave,'tblout','totalDuration'); 
return;

idxuse = strcmp(et.EventType,'INSLeftBatteryLevel') & ...
    ~isempty(et.EventSubType) & ... 
    ~strcmp(et.EventSubType,'%');
etuse = et(idxuse,:); 
percents = cellfun(@(x) str2num( strrep(x,'%','') ),etuse.EventSubType); 
times = etuse.UnixOnsetTime; 
figure;
plot(times,percents); 
title('INS battery decline'); 
xlabel('Time'); 
ylabel('INS %'); 
set(gca,'FontSize',16); 

end