function [Input2] = map(In)
    %����������ά����
    disp('��ʼ������ά����');
    [m,n] = size(In);
    Input2 = [In,zeros(m,1)];
    Dis = zeros(m,1);
    for i = 1:m
        for j = 2:n
            Dis(i) = Dis(i) + Input2(i,j)^2;
        end
    end
    indexMax = find(Dis == max(Dis));
    for i = 1:m
        Input2(i,n+1) = sqrt(max(Dis) - Dis(i));
    end
    disp('������ά�������');
end