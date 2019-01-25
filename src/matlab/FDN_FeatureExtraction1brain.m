function [feature,td] = FDN_FeatureExtraction1brain(Mask,T1,brainregion,featureopen,b,typeoffeature)
%     Input:
    %     T1 (128*256*256): T1 image of a brain MRI
    %     Mask (128*256*256): mark which voxel is chosen to be extracted features 
    %     brainregion: brain region image (0: not brain / 1: brain)
    %     featureopen (=5): size of window openned around the voxel
    %     b (=0.01): replace feature value 0 by 0.01
    %     typeoffeature (=1 / =2): feature 1 cho bai toan cua em, feature 2 cho bai toan cua anh
%     Output:
    %     feature (number of chosen voxel * length of feature vector): fetures
    %     of chosen voxel
    %     td (number of chosen voxel * 3) : toa x,y,z cua cac chosen voxel trong T1

    % Feature
    [x,y,z] = ind2sub(size(T1),find(Mask));    
    si = featureopen*2 + 1;
    if(typeoffeature==1)
        featuresize = si*si*3+3;
    else
        featuresize = si*si*3*3+3;
    end
    feature = zeros(length(x),featuresize);
    label = zeros(length(x),1);
    td = [x y z];
    
    % Normalize grey - devided by mean 
    meangrey = mean(T1(T1>0));
    T1normalized = double(T1)*100/meangrey;
    T1normalized = (double(T1normalized)-100)/200;
%     figure;
%     histogram(T1normalized(T1>0));
%     xlim([1,150]);
    
    
    % Normalize toa do
    [x0,y0,z0] = ind2sub(size(T1),find(brainregion));  
    rangex = [min(x0) max(x0)];
    rangey = [min(y0) max(y0)];
    rangez = [min(z0) max(z0)];
    
    for i=1:length(x)   
%         if(mod(i,10000)==0)
%            i 
%         end
        % Trich feature tung voxel
        feature(i,:) = FeatureExtraction1pix(x(i),y(i),z(i),rangex,rangey,rangez,T1normalized,featureopen,b,typeoffeature);
    end
end

function [feature1pix] = FeatureExtraction1pix(x,y,z,rangexnorm,rangeynorm,rangeznorm,T1,featureopen,b,typeoffeature)
    rangex = [1 size(T1,1)];
    xnorm = double(x-rangexnorm(1))/(rangexnorm(2)-rangexnorm(1)) - 0.5;
    ynorm = double(y-rangeynorm(1))/(rangeynorm(2)-rangeynorm(1)) - 0.5;
    znorm = double(z-rangeznorm(1))/(rangeznorm(2)-rangeznorm(1)) - 0.5;  
        
    if(typeoffeature==1)
        si = featureopen*2 + 1;
        z1slices = T1(max(rangex(1),x-featureopen):min(rangex(2),x+featureopen),y-featureopen:y+featureopen,z-1:z+1);
        if(rangex(1) > x-featureopen)
            phan_bu = zeros(rangex(1)-(x-featureopen),si,3);
            z1slices = [phan_bu;z1slices];
        end
        if(rangex(2) < x+featureopen)
            phan_bu = zeros( (x+featureopen)-rangex(2),si,3); 
            z1slices = [z1slices;phan_bu];      
        end
        
        z1slices(z1slices==0) = b;        
        feature1pix = [reshape(z1slices,[1,si*si*3]),xnorm,ynorm,znorm];    
        
    % Type of feature 2    
    else
        si = featureopen*2 + 1;        
        x3slices = T1(max(rangex(1),x-1):min(rangex(2),x+1),y-featureopen:y+featureopen,z-featureopen:z+featureopen);
        if(rangex(1) > x-1)
            phan_bu = zeros(rangex(1)-(x-1),si,si);
            x3slices = [phan_bu;x3slices];
        end
        if(rangex(2) < x+1)
            phan_bu = zeros( (x+1)-rangex(2),si,si);
            x3slices = [x3slices;phan_bu];            
        end        
        x3slices(x3slices==0) = b;                   
        x01 = x3slices(1,:,:);
        x02 = x3slices(2,:,:);
        x03 = x3slices(3,:,:);
        
        
        
        y3slices = T1(max(rangex(1),x-featureopen):min(rangex(2),x+featureopen),y-1:y+1,z-featureopen:z+featureopen);             
        if(rangex(1) > x-featureopen)
            phan_bu = zeros(rangex(1)-(x-featureopen),3,si);
            y3slices = [phan_bu;y3slices];
        end
        if(rangex(2) < x+featureopen)
            phan_bu = zeros( (x+featureopen)-rangex(2),3,si);
            y3slices = [y3slices;phan_bu]; 
            imshow(squeeze(y3slices(:,2,:)),[]);
        end        
        y3slices(y3slices==0) = b;    
        y01 = y3slices(:,1,:);
        y02 = y3slices(:,2,:);
        y03 = y3slices(:,3,:);        
        
        z3slices = T1(max(rangex(1),x-featureopen):min(rangex(2),x+featureopen),y-featureopen:y+featureopen,z-1:z+1);
        if(rangex(1) > x-featureopen)
            phan_bu = zeros(rangex(1)-(x-featureopen),si,3);
            z3slices = [phan_bu;z3slices];
        end
        if(rangex(2) < x+featureopen)
            phan_bu = zeros( (x+featureopen)-rangex(2),si,3);
            z3slices = [z3slices;phan_bu];      
            imshow(squeeze(z3slices(:,:,2)),[]);
        end        
        z3slices(z3slices==0) = b;         
        z01 = z3slices(:,:,1);
        z02 = z3slices(:,:,2);
        z03 = z3slices(:,:,3);      
        
        feature1pix = [reshape(x01,[1 11*11]),reshape(x02,[1 11*11]),reshape(x03,[1 11*11]),reshape(y01,[1 11*11]),reshape(y02,[1 11*11]),reshape(y03,[1 11*11]),reshape(z01,[1 11*11]),reshape(z02,[1 11*11]),reshape(z03,[1 11*11]),xnorm,ynorm,znorm ];      
    end

end

