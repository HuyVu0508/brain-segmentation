% Low prob + open + CSF
for i=1:12
% for i=1:6
    
    
    % Load file ground truth + T1
    load(strcat('Data DFN/Train/train (',num2str(i),').mat'));
%     load(strcat('Data DFN/Test/test (',num2str(i),').mat'));    
    difference = S.Difference;
    groundtruth_voxel = S.GroundTruth;  
    csf = groundtruth_voxel==1;
    T1 = S.T1;
    
    
    % Load file difference + lowprob + open
    load(strcat('Data DFN/Train notsure + lowprob + open/train (',num2str(i),').mat'));
%     load(strcat('Data DFN/Test notsure + lowprob + open/test (',num2str(i),').mat'));    
    diff_lowprob_open = S.Difference_LowProba_Open;
    
    
    % Combine 
    diff_lowprob_open_csf = diff_lowprob_open|csf;
    S = struct('GroundTruth',groundtruth_voxel,'T1',T1,'Difference_LowProba_Open_CSF',diff_lowprob_open_csf);  
    link_voxel = strcat('Data DFN\train notsure + lowprob + open + CSF');
    save(strcat(link_voxel,'\train (',num2str(i),')'),'S');    
%     link_voxel = strcat('Data DFN\Test notsure + lowprob + open + CSF');
%     save(strcat(link_voxel,'\test (',num2str(i),')'),'S');


    figure;
    subplot(2,2,1);     imshow(difference(:,:,128),[]);
    subplot(2,2,2);     imshow(diff_lowprob_open(:,:,128),[]);
    subplot(2,2,3);     imshow(diff_lowprob_open_csf(:,:,128),[]);

end