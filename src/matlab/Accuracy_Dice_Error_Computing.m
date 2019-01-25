function [pick_perClass_onPredict,pick_all,accuracy_perClass_onPick,accuracy_all_onPick,accuracy_all_on_Label,dice] = Accuracy_Dice_Error_Computing(labelasigned,label,labelasigned2,label2,tol,groundtruth_voxel,result_voxel,numofclass)
%     Input:
    
%     Output:
%     labelasigned: ket qua tra ve tu prediction using max chosen
%     labelasigned2: ket qua tra ve tu prediction su dung nguong tol = [0.8 0.9 0.8];
%     label: label tren toan anh
%     label2: label cua vung chosen su dung nguong tol = [0.8 0.9 0.8];
%     groundtruth_voxel: ground truth 0-1-2-3
%     result_voxel: prediction chosen
    


    numofclass = 3;

   % Accuracy
   % Percentage of right prediction on picked voxels
   accuracy_all_onPick = [sum(labelasigned2==label2) / length(label2)];
   accuracy_perClass_onPick = [];
   pick_perClass_onPredict = [];
   for j = 1:3
       accuracy_perClass_onPick =  [ accuracy_perClass_onPick, sum((labelasigned2==j)&label2==j) / sum(labelasigned2==j)];
       pick_perClass_onPredict = [pick_perClass_onPredict,sum(labelasigned2==j)/sum(labelasigned==j)];
   end
   accuracy_all_on_Label = [sum(labelasigned2==label2) / length(label),sum(labelasigned2~=label2) / length(label)];
   pick_all = length(labelasigned2)/length(labelasigned);   

   
    % Dice & Error
    dice = zeros(1,numofclass);  
    for i=1:numofclass
        groundtruth_voxel_class = groundtruth_voxel==i;
        result_voxel_class = result_voxel==i;
        overlap = (groundtruth_voxel_class&result_voxel_class);        
        % Dice
        dice(i) = sum(overlap(:))*2 / (sum(groundtruth_voxel_class(:))+sum(result_voxel_class(:)));
%         % Error
%         different = (groundtruth_voxel_class(:)~=result_voxel_class(:))&(result_voxel(:)>0);
%         error(i) =  sum(different(:))/sum(groundtruth_voxel_class(:));        
    end    
    overlap = (groundtruth_voxel==result_voxel)&(groundtruth_voxel>0);
    % Dice
    dice(4) = mean(dice([1:3]));    
    % Error
    different = (groundtruth_voxel(:)~=result_voxel(:))&(result_voxel(:)>0);  
    
    
%     % Show results
%     tol
%     pick_perClass_onPredict
%     pick_all
%     accuracy_perClass_onPick
%     accuracy_all_onPick
%     accuracy_all_on_Label    
%     dice
end