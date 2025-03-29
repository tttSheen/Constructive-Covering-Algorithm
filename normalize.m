function [Input] = normalization(In)
%   将数据归一化处理
    disp('开始数据归一化');
    [m,n] = size(In);
    Input = zeros(m,n);
    label = In(:,1);
    data=In(:,2:end);
    max_data=max(data);
    min_data=min(data);
    for i = 1:m
        for j = 1:n-1
            if max_data(j)==min_data(j)
                data(i,j)=0;
            else
                data(i,j) = (data(i,j)-min_data(j))/(max_data(j)-min_data(j));
        end
    end
    
    Input(:,1)=label;
    Input(:,2:end)=data;
end