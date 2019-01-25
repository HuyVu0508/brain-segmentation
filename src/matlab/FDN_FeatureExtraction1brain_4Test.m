function [feature,td] = FDN_FeatureExtraction1brain_4Test(Mask,T1,Label,brainnum,featureopen,b,typeoffeature)
%     Input:
%     T1 (128*256*256): T1 image of a brain MRI
%     Mask (128*256*256): mark which voxel is chosen to be extracted features 
%     featureopen (=5): size of window openned around the voxel
%     b (=0.01): replace feature value 0 by 0.01
%     typeoffeature (=1 / =2): feature 1 cho bai toan cua em, feature 2 cho bai toan cua anh
%     Output:
%     feature (number of chosen voxel * length of feature vector): fetures
%     of chosen voxel
%     td (number of chosen voxel * 3) : toa x,y,z cua cac chosen voxel trong T1

    % Set featuresize
    si = 2*featureopen + 1;
    if(typeoffeature==1)
        featuresize = si*si*3+3;
    else
        featuresize = si*si*3*3+3;
    end
    
    % Feature
    [xbrain,ybrain,zbrain] = ind2sub(size(T1),find(Mask));    
    length(xbrain)
    
    % Normalize grey - devided by mean 
    meangrey = mean(T1(T1>0));
    T1normalized = double(T1)*100/meangrey;
    T1normalized = (double(T1normalized)-100)/200;

    % Tim range de normalize toa do
    rangex = [min(xbrain) max(xbrain)];
    rangey = [min(ybrain) max(ybrain)];
    rangez = [min(zbrain) max(zbrain)];       
    
    % Searching through each slice
    for z=1:size(Mask,3)
       z
       if(max(max(Mask(:,:,z)))>0) 
           slice = Mask(:,:,z);
           [x,y] = ind2sub(size(slice),find(slice));    
%            range(y)
%            range(x)
%            a=1
           length(x);
           feature1slice = zeros(length(x),featuresize);
           label1slice = zeros(length(x),1);
           td1slice = zeros(length(x),3);
           for i=1:length(x)
               xi = x(i);
               yi = y(i);
               [feature1pix] = FeatureExtraction1pix(xi,yi,z,rangex,rangey,rangez,T1normalized,featureopen,b,typeoffeature); 
               label1pix = Label(xi,yi,z);
               td1pix = [xi,yi,z];
               
               feature1slice(i,:) = feature1pix;
               label1slice(i,:) = label1pix;
               td1slice(i,:) = td1pix;
                
           end
           
%            % Write to mat file
%            link = 'Data_DFN\Features\Test\';
%            Xtr0 = feature1slice;    Ytr0 = label1slice;     tdtr
%            save(strcat(link,num2str(Label(xi,yi,z)),'\X\X_no_',num2str(brainnum),'_',num2str(z)),'feature1pix');
%            save(strcat(link,num2str(Label(xi,yi,z)),'\Y\Y_no_',num2str(brainnum),'_',num2str(z)),'label1pix');
%            save(strcat(link,num2str(Label(xi,yi,z)),'\td\td_no_',num2str(brainnum),'_',num2str(z)),'td1pix');           

            % Label 1
            label1 = label1slice==1;
            Xtr1 = feature1slice(label1,:);     Ytr1 = label1slice(label1,:);     tdtr1 = td1slice(label1,:);
            save(strcat('Data_DFN\Features\Test\1\X\X_no_',num2str(brainnum),'_',num2str(z)),'Xtr1','-v7.3');
            save(strcat('Data_DFN\Features\Test\1\Y\Y_no_',num2str(brainnum),'_',num2str(z)),'Ytr1','-v7.3');
            save(strcat('Data_DFN\Features\Test\1\td\td_no_',num2str(brainnum),'_',num2str(z)),'tdtr1','-v7.3');
            % Label 0
            label0 = label1slice==0;
            Xtr0 = feature1slice(label0,:);     Ytr0 = label1slice(label0);     tdtr0 = td1slice(label0,:);
            save(strcat('Data_DFN\Features\Test\0\X\X_no_',num2str(brainnum),'_',num2str(z)),'Xtr0','-v7.3');
            save(strcat('Data_DFN\Features\Test\0\Y\Y_no_',num2str(brainnum),'_',num2str(z)),'Ytr0','-v7.3');
            save(strcat('Data_DFN\Features\Test\0\td\td_no_',num2str(brainnum),'_',num2str(z)),'tdtr0','-v7.3');       
       end
    end
end

function [feature1pix] = FeatureExtraction1pix(x,y,z,rangexnorm,rangeynorm,rangeznorm,T1,featureopen,b,typeoffeature)
    rangex = [1 size(T1,1)];
    xnorm = double(x-rangexnorm(1))/(rangexnorm(2)-rangexnorm(1));
    ynorm = double(y-rangeynorm(1))/(rangeynorm(2)-rangeynorm(1));
    znorm = double(z-rangeznorm(1))/(rangeznorm(2)-rangeznorm(1));  
        
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
        
        feature1pix = [reshape(x3slices,[1,3*si*si]),reshape(y3slices,[1,3*si*si]),reshape(z3slices,[1,3*si*si]),xnorm,ynorm,znorm];   
    end

end

