function clr = Colorize(V)

    % colorize
    clr = zeros([size(V),3]);
    idx = find(V(:)==1);    
    for i=1:length(idx)
        [x,y] = ind2sub(size(V),idx(i));
        size(clr)
        clr(x,y,:) = [1 0 0];    
        
    end
    idx = find(V(:)==2);    
    for i=1:length(idx)
        [x,y] = ind2sub(size(V),idx(i));
        clr(x,y,:) = [0 1 0];      
    end
    idx = find(V(:)==3);    
    for i=1:length(idx)
        [x,y] = ind2sub(size(V),idx(i));
        clr(x,y,:) = [0 0 1];       
    end

end