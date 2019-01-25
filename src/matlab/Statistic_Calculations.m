s = 0;
for i=1:6
    figure;
    
%     load(strcat('Send anh Huy\Train notsure + lowprob + open\',num2str(i),'\train (',num2str(i),').mat'));
%     new = sum(S.Open_not_LowPro(:));
%     imshow(S.Open_not_LowPro(:,:,128),[]);

%     load(strcat('Data DFN\Train notsure + lowprob + open\train (',num2str(i),').mat'));
%     new = sum(S.Difference_LowProba_Open(:));
%     imshow(S.Difference_LowProba_Open(:,:,128),[]);
    
%     load(strcat('Data DFN\Train\train (',num2str(i),').mat'));
%     new = sum(S.Difference(:));
%     imshow(S.Difference(:,:,128),[]);    
    
    
%     load(strcat('Data DFN\Train notsure + lowprob\train (',num2str(i),').mat'));
%     new = sum(S.Difference_LowProb(:));
%     imshow(S.Difference_LowProb(:,:,128),[]);    
    
%     load(strcat('Data DFN\Train notsure + lowprob + open + CSF\train (',num2str(i),').mat'));
%     new = sum(S.Difference_LowProba_Open_CSF(:));
%     imshow(S.Difference_LowProba_Open_CSF(:,:,128),[]);
%     load(strcat('Data DFN\Test\test (',num2str(i),').mat'));
%     new = sum(S.Difference(:));
%     imshow(S.Difference(:,:,128),[]);    

    s = s + new;
end
s