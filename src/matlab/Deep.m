%% Choose task and deep
% Choose task
Task = 'Train';
Task = 'Test';

% Choose deep 
Deep = 1;
Deep = 2;

%% Loading data
% Train Deep 1
if( strcmp(Task,'Train') && Deep==1 )
    link = '\Train\Data\RawData.mat';
    load(link);
    RawData = RawData;    
    numofdata = length(EMGMWrong);
    link = '\Train\Data\EMGMWrong.mat';
    load(link);
    EMGMWrong = EMGMWrong;
    linktrain = '\Train\Data\Label';
    load(link);
    Label = Label;
end

% Train Deep 2
if( strcmp(Task,'Train') && Deep==2 )
    link = '\Train\Data\RawData.mat';
    load(link);
    RawData = RawData;    
    numofdata = length(EMGMWrong);
    link = '\Train\Data\EMGMWrong.mat';
    load(link);
    EMGMWrong = EMGMWrong;
    linktrain = '\Train\Data\Label';
    load(link);
    Label = Label;
end

% Test Deep 1
if( strcmp(Task,'Test') && Deep==1 )
    link = '\Test\Data\RawData';
    load(link);
    RawData = RawData;    
    numofdata = length(EMGMWrong);
end

% Test Deep 2
if( strcmp(Task,'Test') && Deep==2 )
    link = '\Test\Data\RawData';
    load(link);
    RawData = RawData;    
    numofdata = length(EMGMWrong);
    link = '\Test\Data\UnsureVoxels';
    load(link);
    UnsureVoxels = UnsureVoxels;        
end

disp('Loading DONE');

%% Feature extraction
feature = {};
label = {};
index = {};
sumnumofpositive = 0;
for i=1:numofdata

    % Extracting Mask
    % Train
        % Deep 1
        if( strcmp(Task,'Train') && Deep==1 )
            % Extract Mask & label1brain
            1EMGMWrong = EMGMWrong{i};
            1Label = Label{i};
            1RawData = RawData{i};
            
            positiveMask = 1EMGMWrong;
            numofpositive = sum(1EMGMWrong(:));
            negativeMask = (1Label>0)~=positiveSamples;
            aray = find(negativeMask0); 
            randomNegativeSelect = datasample(aray,floor(numofpositive*1.5),'Replace',false);
            sumnumofpositive = sumnumofpositive + numofpositive;
            negativeMask = zeros(size(1Label));
            negativeMask(randomNegativeSelect) = 1;
            Mask = positiveMask|negativeMask;      
            label1brain = 1EMGMWrong(Mask);
        end
        
        % Deep 2
        if( strcmp(Task,'Train') && Deep==2 )
            % Extract Mask & label1brain
            1EMGMWrong = EMGMWrong{i};
            1Label = Label{i};
            1RawData = RawData{i}; 
            Mask = 1EMGMWrong; 
            label1brain = 1Label(Mask);
        end
        
    % If Test
        % Deep 1 
        if( strcmp(Task,'Test') && Deep==1 )
            1RawData = RawData{i};
            Mask = RawData{i}>0;           
        end        

        % Deep 2
        if( strcmp(Task,'Test') && Deep==2 )
            1RawData = RawData{i};
            Mask = UnsureVoxels{i};
        end     
        
    % Extracting feature & indx
    featureopen = 5;
    b = 0.01;
    typeoffeature = 2;
    [feature1brain,index1brain] = Deep_FeatureExtraction_WholeBrain(Mask,1RawData,featureopen,b,typeoffeature);
    feature = [feature,feature1brain];
    index = [index, index1brain];
    
    % Extracting label (for Train only)
    if(strcmp(Task,'Train'))
        label = [label, label1brain]; 
    end
end

%% Save to .mat file
% Train Deep 1
if( strcmp(Task,'Train') && Deep==1 )
    % Save [feature,index,label]
    % Save into Train\Data\Deep1Train
end

% Train Deep 2
if( strcmp(Task,'Train') && Deep==2 )
    
end

% Test Deep 1
if( strcmp(Task,'Test') && Deep==1 )
    % Save [feature,index]
    % Save into Train\Data\Deep1Test
end

% Test Deep 2
if( strcmp(Task,'Test') && Deep==2 )
    
end