function feature = Image_Feature_Extraction_1brain(Voxel)
    load('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\Data Original\Training\Train\train (8).mat')
    Voxel = S.Voxel(S.Voxel>0);
    histogram(Voxel(:),'BinLimits',[0,255],'BinWidth',1);
    % Omit the abnormal
        





end
