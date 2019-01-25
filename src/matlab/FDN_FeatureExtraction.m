function [feature0,label0,indx0] = FDN_FeatureExtraction(data,numofdata,featureopen,b,typeoffeature)
    feature0 = {};
    label0 = {};
    indx0 = {};
    sumnumofpositive = 0;
    for i=1:numofdata
%     for i=1:1
        tic
        i
        
        % Positive samples
%         positiveSamples = data{i}.S.Difference;
        positiveSamples = data{i}.S.LowProba;
%         positiveSamples = data{i}.S.Difference_LowProba_Open_CSF;
        
        % Extract positive and negative masks
        positiveMask = positiveSamples;
        numofpositive = sum(positiveMask(:));
        
        negativeMask0 = (data{i}.S.GroundTruth>0)~=positiveSamples;
        aray = find(negativeMask0); 
        rand_neg_ele = datasample(aray,floor(numofpositive*1.5),'Replace',false);
        sumnumofpositive = sumnumofpositive + numofpositive;
        negativeMask = zeros(size(data{i}.S.GroundTruth));
        negativeMask(rand_neg_ele) = 1;
        
        Mask = positiveMask|negativeMask;
        T1 = data{i}.S.T1;    
        label = positiveSamples(Mask);
        
        [ft,td] = FDN_FeatureExtraction1brain(Mask,T1,featureopen,b,typeoffeature);
        feature0 = [feature0,ft];
        label0 = [label0, label];    
        indx0 = [indx0, td];
        toc
    end
    
    sumnumofpositive
end
