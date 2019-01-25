%% Load data
disp('Loading data ...')

linktrain = 'C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\Data DFN\Train';
list = dir(strcat(linktrain,'\train *'));
numoftrain = size(list,1);
datatrain = {};
for i=1:numoftrain
    datatrain{i} = load(strcat(linktrain,'\train (',num2str(i),')'));
end

linktest = 'C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\Data DFN\Test';
list = dir(strcat(linktest,'\test *'));
numoftest = size(list,1);
datatest = {};
for i=1:numoftest
    datatest{i} = load(strcat(linktest,'\test (',num2str(i),')'));
end

disp('Loading DONE');

%% Feature extraction
disp('Extracting features ...')

% Train data
[featuretrain0,labeltrain0,indxtrain0] = FDN_FeatureExtraction(datatrain,numoftrain,5,0.01,2);
save('Data DFN\Features\Train\feature_label_indx_train.mat','featuretrain0','labeltrain0','indxtrain0','-v7.3');

% Test data
 FDN_FeatureExtraction_4Test(datatest,numoftest,5,0.01,1);

% Save data
pick = 1:4;
featuretrain = cell2mat(featuretrain0(pick));
labeltrain = cell2mat(labeltrain0(pick));
indxtrain = cell2mat(indxtrain0(pick));
% Label 1
label1 = labeltrain==1;
Xtr1 = featuretrain(label1,:);     Ytr1 = labeltrain(label1,:);     tdtr1 = indxtrain(label1,:);
save('Data_DFN\Features\Train\1\X\X_train_1_4','Xtr1','-v7.3');
save('Data_DFN\Features\Train\1\Y\Y_train_1_4','Ytr1','-v7.3');
save('Data_DFN\Features\Train\1\td\td_train_1_4','tdtr1','-v7.3');
% Label 0
label0 = labeltrain==0;
Xtr0 = featuretrain(label0,:);     Ytr0 = labeltrain(label0);     tdtr0 = indxtrain(label0,:);
save('Data_DFN\Features\Train\0\X\X_train_1_4','Xtr0','-v7.3');
save('Data_DFN\Features\Train\0\Y\Y_train_1_4','Ytr0','-v7.3');
save('Data_DFN\Features\Train\0\td\td_train_1_4','tdtr0','-v7.3');

disp('Feature extraction DONE');

%% Training Deep
% Training 
featuretrain = cell2mat(featuretrain0);
labeltrain = cell2mat(labeltrain0);

%% Testing
% Testing
featuretest = cell2mat(featuretest0);
labeltest = cell2mat(labeltest0);