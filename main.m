clear
clc
%Ԥ�����load dataset
load('./dataSet/iris.mat')
%load('./dataSet/balance-scale.mat')
%load('./dataSet/lymphography.mat')
%load('./dataSet/ionosphere.mat')
input=normalize(DataSet);%��һ��
input2=map(input);%��ά
[m,n]=size(input2);
indics=crossvalind('Kfold',m,10);%���ֽ�����֤��
%%
rate_ave_sum=0;
%����һ����ȡƽ��
for g=1:100
    rate_sum=0;
for round=1:10
    test=(indics==round);
    test_data=input2(test,2:n);
    test_label=input2(test,1);
    train=~test;
    train_data=input2(train,2:n);
    train_label=input2(train,1);
    %%
    %training
    %disp('start training...');
    output=[];
    len=length(train_data);
    train_learn=zeros(len,1);%0��ʾδѧϰ��1��ʾ��ѧϰ
    unlearn=find(train_learn==0);%δѧϰ��������
    while unlearn
        ulen=length(unlearn);
        k=ceil((ulen-1)*rand(1)+1);%�����ѡһ��δѧϰ����,unlearn(k)Ϊ��������train_data�е�����
        xk=train_data(unlearn(k),:);%��Ϊ��������
        xk_label=train_label(unlearn(k));
        diff_class=unlearn(find(train_label(unlearn)~=xk_label));%����δѧϰ��������
        same_class=unlearn(find(train_label(unlearn)==xk_label));%ͬ��δѧϰ��������
        diff_dist=[];same_dist=[];
        for i=1:length(diff_class)
            diff_dist(i,1)=sum(train_data(diff_class(i),:).*xk);%���������ڻ�
        end
        for i=1:length(same_class)
            same_dist(i,1)=sum(train_data(same_class(i),:).*xk);%ͬ�������ڻ�
        end
        if isempty(diff_class)&&~isempty(same_class)
            r=min(same_dist);%��ֻ��ͬ����������������������ȡͬ�����������������ĵ�������Ϊ�뾶����С�ڻ���
        elseif isempty(same_class)&&~isempty(diff_class)
            r=max(diff_dist)*2;%��ֻ��������������ͬ����������ȡ�������������������ĵ���С�����һ��Ϊ�뾶������ڻ���
        elseif isempty(same_class)&&isempty(diff_class) 
            r=xk.*xk;%������ͬ������Ҳ��������������ȡ�����������Լ����ڻ�Ϊ�뾶
        else
            min_diff=max(diff_dist);%�����������
            r=min(same_dist(find(same_dist>min_diff)));%��С�뾶��,����Զͬ���
        end
        output_temp.center=unlearn(k);
        output_temp.cdata=xk;
        output_temp.label=xk_label;
        output_temp.r=r;
        output_temp.cnt=1;
        output_temp.samples={unlearn(k)};
        train_learn(unlearn(k))=1;%�������ı��Ϊ��ѧϰ
        for i=1:length(same_dist)
            if same_dist(i)>r
                if same_class(i)~=unlearn(k)
                    output_temp.samples=[output_temp.samples,same_class(i)];
                    train_learn(same_class(i))=1;%���Ϊ��ѧϰ
                    output_temp.cnt=output_temp.cnt+1;
                end
            end
        end
        output=[output;output_temp];
        unlearn=find(train_learn==0);%����δѧϰ��������      
    end
    sum([output.cnt]);
    %%
    %testing
    %disp('start testing...');
    test_temp=[];result_label=[];test_dist=[];
    [l,w]=size(test_data);
    for t=1:l
        for j=1:length(output)
            test_dist(t,j)=sum(test_data(t,:).*output(j).cdata);%������Լ�������ÿ���������ĵ��ڻ�
            if test_dist(t,j)>=output(j).r
                test_temp(t,j)=1;%test_temp�߼�ֵ��=1��ʾ�ڵ�ǰ������
            else
                test_temp(t,j)=0;%=0��ʾ���ڵ�ǰ������
            end
        end
    end
    test_temp(:,j+1)=sum(test_temp,2);%��test_temp���
    for t=1:length(test_temp(:,j+1))
        if test_temp(t,j+1)==1%��sum(test_temp)=1�����ʾ������ֻ����һ�������ڣ����仮�ֵ��˸��Ƕ�Ӧ�������!=1�����ӳپ���
            index=find(test_temp(t,1:end-1)==1);
            result_label(t,1)=output(index).label;%result_label��һ�б�ʾ�ӳپ���ǰ�Ļ��ֽ��
            result_label(t,2)=output(index).label;%�ڶ��б�ʾ�ӳپ��ߺ�Ļ��ֽ��
            %�ӳپ���
        elseif test_temp(t,j+1)==0 %=0��ʾ�������κ�һ�����ǣ���ʱ���������ֵ���������ĸ�����
            maxdist=max(test_dist(t,:));
            index=find(test_dist(t,:)==maxdist);
            result_label(t,1)=-1;%���һ��
            result_label(t,2)=output(index).label;
        else%��ʾ������������
            ind=find(test_temp(t,:)==1);
            maxdist=max(test_dist(t,ind));
            index=find(test_dist(t,:)==maxdist);
            result_label(t,1)=0;%���һ��
            result_label(t,2)=output(index).label;           
        end           
    end
    %disp(['result of round ',num2str(round),':']);
    %disp(result_label(:,2)');
    %disp(['class of round',num2str(round),':']);
   % disp(test_label');    
    right=sum(result_label(:,2)==test_label);
    whole=length(test_label);
    rate=right/whole;
    %disp(['������ȷ����',num2str(right)]);
    %disp(['********��ȷ��**********��',num2str(rate)]);
    rate_sum=rate+rate_sum;
end
    rate_ave=rate_sum/10;
    accuracy_ave(g,1)=rate_ave;
    disp(['rate_average per round:',num2str(rate_ave)]);
    rate_ave_sum=rate_ave_sum+rate_ave;
end
rate_ave_sum=rate_ave_sum/100;
disp(['**********************rate_average all rounds:',num2str(rate_ave_sum)]);
disp(['**********************����Ϊ��',num2str(var(accuracy_ave))]);
