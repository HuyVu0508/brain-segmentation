function [featureall,labelall,indxall] = EMGM_FeatureExtraction(data,numofdata)

% Feature - label extraction
featureall = cell(numofdata,1);
labelall = cell(numofdata,1);
indxall = cell(numofdata,1);
for i=1:numofdata
    i
    feature = {};
    label = {};
    indx = {};    
    
    T1 = data{i}.S.Voxel;
    groundtruth = data{i}.S.GroundTruth;
    [feature0, label0,indx0] = EmGmFeatureExtraction(T1,groundtruth); 
    feature0;
    
    featureall{i} = feature0;
    labelall{i} = label0;
    indxall{i} = indx0;

    
    % Detect the abnormal
    abnormal_indx = (feature0<10) & (label0==2);
    sum(abnormal_indx);
    abnormal_onImg_indx = indx0(abnormal_indx);
    [x y z] = ind2sub(size(T1),abnormal_onImg_indx);
    z_uni = unique(z)';
%     for zi=z_uni
%         xz = x(z==zi);
%         yz = y(z==zi);
%         figure;
%         img_abnormal = T1(:,:,zi);
%         for j=1:length(xz)
%             img_abnormal(xz(j),yz(j)) = 0;
%         end        
%         imshow(img_abnormal,[]);
%         title(strcat('Number -  ',num2str(i)));
%     end
   
    
end   
end

function [feature, label,indx] = EmGmFeatureExtraction(brain,groundtruth)
% Input:
% brain: T1 image
% groundtruth
% Output:
% % feature: list of feature of each voxels
% label: nhu tren
% indx: index of each voxel in the original 3D

indx = find(groundtruth);
label = groundtruth(indx);

[x,y,z] = ind2sub(size(groundtruth),indx);
feature=zeros(length(indx),1);
% Normalize
norm_factor = mean(brain(indx));
norm_brain = double(brain)*100 / norm_factor;

for i=1:length(indx)
    i;        
    feature(i,1) = norm_brain(x(i),y(i),z(i));
%     feature(i,1) = brain(x(i),y(i),z(i));
%     feature(i,2) = mean([brain(x(i)+1,y(i),z(i)),brain(x(i)-1,y(i),z(i)),brain(x(i),y(i)+1,z(i)),brain(x(i),y(i)-1,z(i))]);
%     feature(i,2:3) = [x(i),y(i)];   
%     feature(i,2) = norm([x(i),y(i),z(i)] - center);
end
end
