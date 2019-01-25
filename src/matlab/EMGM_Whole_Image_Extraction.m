%% Function: extracting feature for 1 image (number of voxels of type 1, type 2 and type 3 brain cells)
% Input:

% Output:

%% Load data

disp('Loading data ...')
% Load data nii
linktrain = 'Data Original\Training\Train';
list = dir(strcat(linktrain,'\train *'));
numoftrain = size(list,1);
datatrain = {};
for i=1:numoftrain
    datatrain{i} = load(strcat(linktrain,'\train (',num2str(i),')'));
end

linktest = 'Data Original\Testing';
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
% [featuretrain0,labeltrain0,indxtrain0] = EMGM_FeatureExtraction(datatrain,numoftrain);
[featuretrain0,labeltrain0,indxtrain0] = EMGM_FeatureExtraction_for_whole_image(datatrain,numoftrain);

% Test data
[featuretest0,labeltest0,indxtest0] = EMGM_FeatureExtraction(datatest,numoftest);

disp('Feature extraction DONE');

%% EM - GM Traning
disp('Traning EM-GM ...');

featuretrain = cell2mat(featuretrain0);
labeltrain = cell2mat(labeltrain0);
indxtrain = cell2mat(indxtrain0);

% % % Loai voxel qua lon @@
% chosen = featuretrain(:,1)<300;
% featuretrain = featuretrain(chosen);
% labeltrain = labeltrain(chosen);


% Shuffle
shuffle = randperm(length(featuretrain));
featuretrain = featuretrain(shuffle,:);
labeltrain = labeltrain(shuffle,:);
indxtrain = indxtrain(shuffle,:);

% Omitting feature<5
omit = featuretrain(:,1)<5 & labeltrain==2;
featuretrain_omit = featuretrain;       featuretrain_omit(omit,:)=[];
labeltrain_omit = labeltrain;           labeltrain_omit(omit,:) = [];
indxtrain_omit = indxtrain;             indxtrain_omit(omit,:) = [];

% Ploting histogram
for i=1:3
    figure;
class_chosen = labeltrain_omit==i;
histogram(featuretrain_omit(class_chosen,1),130);
ylim([0,12*10^4]);
% xlim([0 1]);
end

% Training
addpath('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\EmGm\EmGm');
[aaa,model,llh] = mixGaussEm(double(featuretrain_omit(:,1)'),double(labeltrain_omit'));    
% [aaa,model,llh] = mixGaussEm(double(featuretrain'),);    

figure; 
plot(llh); 
save('EMGM results\12 trains\model.mat','model');

disp('Training DONE');

%% Testing

disp('Testing ...');
% Saving directory
namemodel = 'grey_maxprob_snormalized';
link_save_result = strcat('Validation testing results\',namemodel);
% Variables
featuretest = cell2mat(featuretest0);
labeltest = cell2mat(labeltest0);
result_voxel_all = {};
resultall = {};
model_cluster = {};
sum_proba = {}; 


for i=1:numoftest

   %% Clustering
    addpath('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\EmGm\EmGm');   
    % Omitting feature<5 for training
    omit = featuretest0{i}<5;
    featuretest0_omit = featuretest0{i};    % labeltest0_omit = labeltest0{i};
    featuretest0_omit(omit)=[];             % labeltest0_omit(omit)=[];
   [~,model2,llh] = mixGaussEm(double(featuretest0_omit'),model);
   figure; plot(llh);   
   model_cluster = [model_cluster, model2];
   % Prediction for all voxels (not omit anything)
   [labelasigned,prob_cluster ] = mixGaussPred((featuretest0{i})',model2);
   disp('Clustering DONE');
   feature = featuretest0{i};
   meanclass = [ mean(feature(labelasigned==1)), mean(feature(labelasigned==2)), mean(feature(labelasigned==3))];
   truelabel = [[1;2;3] [0;0;0]];
   [~,truelabel(1,2)] = min(meanclass);
   [~,truelabel(3,2)] = max(meanclass);
   truelabel(2,2) = find(meanclass<max(meanclass) & meanclass>min(meanclass));
   unique(labelasigned)
   
   
   % Set true label
   labelasigned_cluster = zeros(size(labelasigned));
   for j=1:length(labelasigned)
        labelasigned_cluster(j) = truelabel(labelasigned(j),2);
   end
  
    % Treat to special voxel (voxels that we omit when training)
    special = featuretest0{i}<5;
    labelasigned_cluster(special)=(zeros(1,sum(special))+2);
  
   %% Omitting low proba voxels
   for j=1:length(labelasigned_cluster)
       if( max(prob_cluster(j,:))<0.7)
            labelasigned_cluster(j) = 0;
       end
   end
   
   % Ploting histogram
   figure;  
   hold on;
    for k=1:3
        class_chosen = (labelasigned_cluster==k);
%         class_chosen = (labelasigned==k);
%         class_chosen = (labeltest0{i}==k);  
        histogram(featuretest0{i}(class_chosen',1),[0: 2: 200]);
    end
    hold off;
   
   %% Reconstruct to 3D brain
   indx2 = indxtest0{i};
   groundtruth_voxel = datatest{i}.S.GroundTruth;
   result_voxel = zeros(size(groundtruth_voxel));
   result_voxel(indx2) = labelasigned_cluster;
   
%    for j=100:150
%       figure;
%       imshow(Colorize(result_voxel(:,:,j)),[]);
%    end
   
    %% Save resuls for later extracting quantity feature for imagest
    result_voxel_all = [result_voxel_all,result_voxel];

end

%% Save resuls for later extracting quantity feature for imagest
    save(strcat('EMGM results\12 trains\training features'),'result_voxel_all');   
   