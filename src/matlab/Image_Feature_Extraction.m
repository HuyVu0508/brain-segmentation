function feature = Image_Feature_Extraction(img)
    load('C:\Users\ThanhHuy\Desktop\Materials\Projects\Brain MRI\Matlab Code\Data Original\Training\Train\train (1).mat')
    img = S.Voxel(:,:,200);
%     figure;
    
    
    % Feature 1 & Feature 2
    [row,col] = find(img);
    feature1 = max(row) - min(row)
    feature2 = max(col) - min(col)
    center = [ (max(row) + min(row))/2 , (max(col) + min(col))/2 ];
    
    
    % Feature gradient
%     BW = edge(img,'Sobel');
%     imshow(BW);
    [Gmag, Gdir] = imgradient(img,'sobel');
%     imtool(Gmag);
    num = 8;    % Divide 360 degree into 8 parts
    box = zeros(1,num);
    angle = 360/num;
    
    [x,y] = find(Gmag);
    for i = 1:length(x)
       vector = [x(i),y(i)] - center;
       if(norm(vector)>0)
            angle_vector = acos(vector*[1 0]'/norm(vector))/(2*pi)*360;
       end
       num_vector = floor(angle_vector/angle) + 1;
       if(vector(2)<=0)
            box(num_vector) = box(num_vector) + 1;       
       else
            box(num_vector + num/2) = box(num_vector + num/2) + 1;  
       end
    end    
    feature3 = box;
    
    
    % Feature EMGM moutains
    
    
    % Feature combination
    feature = [feature1, feature2, feature3];
    

end