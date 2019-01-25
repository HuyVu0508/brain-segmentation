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
[featuretrain0,labeltrain0,indxtrain0] = EMGM_FeatureExtraction(datatrain,numoftrain);
% [featuretrain0,labeltrain0,indxtrain0] = EMGM_FeatureExtraction_2(datatrain,numoftrain);

% % Test data
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
histogram(featuretrain_omit(class_chosen,4),130);
ylim([0,12*10^4]);
xlim([0 1]);
end

% Training
addpath('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\EmGm\EmGm');
[aaa,model,llh] = mixGaussEm(double(featuretrain_omit'),double(labeltrain_omit'));    
% [aaa,model,llh] = mixGaussEm(double(featuretrain'),);    

figure; 
plot(llh); 

disp('Training DONE');

%% Testing

disp('Testing ...');
% Saving directory
namemodel = 'grey_maxprob_snormalized';
link_save_result = strcat('Validation testing results\',namemodel);
% if(exist(link_save_result)~=7)
%     mkdir(link_save_result);
% else
%     disp('Folder already exist');
%     stoperror;
% end

% Variables
featuretest = cell2mat(featuretest0);
labeltest = cell2mat(labeltest0);
result_voxelall = {};
resultall = {};
model_cluster = {};
sum_proba = {}; 

for i=1:numoftest
% for i=3:3

    addpath('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\EmGm\EmGm');
%    % Choosing high prob elements
   [labelasigned_train,prob_train] = mixGaussPred((featuretest0{i})',model);

   % Clustering
    % Omitting feature<5 for training
    omit = featuretest0{i}<5;
    featuretest0_omit = featuretest0{i};    % labeltest0_omit = labeltest0{i};
    featuretest0_omit(omit)=[];             % labeltest0_omit(omit)=[];
   [~,model2,llh] = mixGaussEm(double(featuretest0_omit'),model);
   model_cluster = [model_cluster, model2];
   % Testing for all voxels (not omit anything)
   [labelasigned,prob_cluster ] = mixGaussPred((featuretest0{i})',model2);
   figure; plot(llh);
   disp('Clustering DONE');
   feature = featuretest0{i};
   meanclass = [ mean(feature(labelasigned==1)), mean(feature(labelasigned==2)), mean(feature(labelasigned==3))];
   truelabel = [[1;2;3] [0;0;0]];
   [~,truelabel(1,2)] = min(meanclass);
   [~,truelabel(3,2)] = max(meanclass);
   truelabel(2,2) = find(meanclass<max(meanclass) & meanclass>min(meanclass));
   
   % Sum probability
   prob_two = (prob_cluster + prob_train)/2;
   sum_proba = [sum_proba, prob_two];
   
   % Set true label
   labelasigned_cluster = zeros(size(labelasigned));
   for j=1:length(labelasigned)
        labelasigned_cluster(j) = truelabel(labelasigned(j),2);
   end
   
   
   % 2 methods combined
   for j=1:length(labelasigned)
       condition1 = labelasigned_train(j)==2;
       condition2 = labelasigned_cluster(j)==3;
       if(condition1)
           labelasigned(j) = labelasigned_train(j);      
       elseif(condition2)
           labelasigned(j) = labelasigned_cluster(j);
       else
           [~,labelasigned(j)] =  max(prob_train(j,:) + prob_cluster(j,:));
       end
       
       if(condition1 & condition2)
           [~,labelasigned(j)] =  max(prob_train(j,:) + prob_cluster(j,:));
       end          
   end  
%    % combine way 2
%    for j=1:length(labelasigned)
%         [~,labelasigned(j)] =  max(prob_train(j,:) + prob_cluster(j,:));
%    end
   
    % Treat to special voxel
    special = featuretest0{i}<5;
    labelasigned(special)=(zeros(1,sum(special))+2);
  
   % Traditional way - max prob
   tol = [0 0 0];
   labelasigned2 = labelasigned;
   labeltest2 = labeltest0{i}';
   indx2 = indxtest0{i};
   
   % Reconstruct prediction voxel 
   groundtruth_voxel = datatest{i}.S.GroundTruth;
   result_voxel = zeros(size(groundtruth_voxel));                
   result_voxel(indx2) = labelasigned2;
   result_voxelall = [result_voxelall,result_voxel];   
%    figure; imshow(groundtruth_voxel(:,:,128),[0 3]);
%    figure; imshow(result_voxel(:,:,128),[0 3]);
%    close all;
   
   % Print images 
%     [~,~,z0] = ind2sub(size(groundtruth_voxel),indxtest0{i}(1));
%     [~,~,z1] = ind2sub(size(groundtruth_voxel),indxtest0{i}(length(indxtest0{i})));  
%     mkdir(strcat(link_save_result,'\',num2str(i)));
%     for j=z0:z1
%         figure('visible','off');
%         subplot(1,2,1);     imshow(groundtruth_voxel(:,:,j),[0 3]);      title('Ground truth');
%         subplot(1,2,2);     imshow(result_voxel(:,:,j),[0 3]);           title('Prediction');
%         print(strcat(link_save_result,'\',num2str(i),'\',num2str(j)),'-dpng');
%     end

   % Calculating accuracy, dice, error
   [pick_perClass_onPredict,pick_all,accuracy_perClass_onPick,accuracy_all_onPick,accuracy_all_on_Label,dice] = Accuracy_Dice_ErrorComputing(labelasigned,labeltest0{i},labelasigned2,labeltest2,tol,groundtruth_voxel,result_voxel,3);
   
   % Save results all
   result = {pick_perClass_onPredict,pick_all,accuracy_perClass_onPick,accuracy_all_onPick,accuracy_all_on_Label,dice};
   resultall = [resultall,{result}];   
    
%    close all;   
   
end

% Average statistics
disp('------- Average statistics -------');
pick_perClass_onPredict = [];
pick_all = [];
accuracy_perClass_onPick = [];
accuracy_all_onPick = [];
accuracy_all_on_Label = [];
dice = [];
for i=1:length(resultall)
    pick_perClass_onPredict =[pick_perClass_onPredict;resultall{i}{1}];
    pick_all = [pick_all;resultall{i}{2}];
    accuracy_perClass_onPick = [accuracy_perClass_onPick;resultall{i}{3}];
    accuracy_all_onPick = [accuracy_all_onPick;resultall{i}{4}];
    accuracy_all_on_Label = [accuracy_all_on_Label;resultall{i}{5}];
    dice = [dice;resultall{i}{6}];  
end
tol
pick_perClass_onPredict = mean(pick_perClass_onPredict)
pick_all = mean(pick_all)
accuracy_perClass_onPick = mean(accuracy_perClass_onPick)
accuracy_all_onPick = mean(accuracy_all_onPick)
accuracy_all_on_Label = mean(accuracy_all_on_Label)
dice = [mean(dice);std(dice) ]

%% Saving EMGM results

% for i=1:numoftest
%    EMGM_result = result_voxelall{i}; 
%    link = 'Test System\EMGM results';
%    save(strcat(link,'\EMGM_result_',num2str(i)),'EMGM_result','sum_proba');
% end
link = 'EMGM results\12 trains';
save(strcat(link,'\EMGM_result'),'result_voxelall','sum_proba');

%% Extracting groundtruth x prediction

for i=1:numoftest
    
    % Wrong pixels
    groundtruth_voxel = datatest{i}.S.GroundTruth;
    result_voxel = result_voxelall{i};
    wrong = groundtruth_voxel~=result_voxel;
    
    % Reconstruct low probability pixels
   indx2 = indxtest0{i};
   low_prob_voxel = zeros(size(wrong));   
   tol = 0.7;   
   low_prob = max(sum_proba{i},[],2)<tol;
   low_prob_voxel(indx2) = low_prob;    
    notsure = wrong|low_prob_voxel;
    right_lowprob_voxel = (~wrong) & low_prob_voxel;
    
    % Open wrong:
    % I dont remember where this code is
    
    % Add CSF:
    % In FDN2_Label_Extraction.m

    % Save:
    link_voxel = 'C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\Data DFN\Train notsure + lowprob + open';
%     save(strcat(link_voxel,'\train (',num2str(i),')'),'S');



end
