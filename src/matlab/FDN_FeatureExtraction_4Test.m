function  FDN_FeatureExtraction_4Test(data,numofdata,featureopen,b,typeoffeature)

    for i=1:numofdata
       tic
       i       
       Mask = (data{i}.S.GroundTruth>0);
       Label = data{i}.S.Difference;
       T1 = data{i}.S.T1;
       FDN_FeatureExtraction1brain_4Test(Mask,T1,Label,i,featureopen,b,typeoffeature);   
       toc 
    end
end

