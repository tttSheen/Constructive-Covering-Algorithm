clear
clc
%预处理后load dataset
load('./dataSet/iris.mat')
%load('./dataSet/balance-scale.mat')
%load('./dataSet/lymphography.mat')
%load('./dataSet/ionosphere.mat')
input=normalize(DataSet);%归一化
input2=map(input);%升维
[m,n]=size(input2);
indics=crossvalind('Kfold',m,10);%划分交叉验证集
%%
rate_ave_sum=0;
%运行一百轮取平均
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
    train_learn=zeros(len,1);%0表示未学习，1表示已学习
    unlearn=find(train_learn==0);%未学习样本索引
    while unlearn
        ulen=length(unlearn);
        k=ceil((ulen-1)*rand(1)+1);%随机挑选一个未学习样本,unlearn(k)为该样本在train_data中的索引
        xk=train_data(unlearn(k),:);%作为覆盖中心
        xk_label=train_label(unlearn(k));
        diff_class=unlearn(find(train_label(unlearn)~=xk_label));%异类未学习样本索引
        same_class=unlearn(find(train_label(unlearn)==xk_label));%同类未学习样本索引
        diff_dist=[];same_dist=[];
        for i=1:length(diff_class)
            diff_dist(i,1)=sum(train_data(diff_class(i),:).*xk);%异类样本内积
        end
        for i=1:length(same_class)
            same_dist(i,1)=sum(train_data(same_class(i),:).*xk);%同类样本内积
        end
        if isempty(diff_class)&&~isempty(same_class)
            r=min(same_dist);%若只有同类样本，无异类样本，则取同类样本中与样本中心的最大距离为半径（最小内积）
        elseif isempty(same_class)&&~isempty(diff_class)
            r=max(diff_dist)*2;%若只有异类样本，无同类样本，则取异类样本中与样本中心的最小距离的一半为半径（最大内积）
        elseif isempty(same_class)&&isempty(diff_class) 
            r=xk.*xk;%若既无同类样本也无异类样本，则取样本中心与自己的内积为半径
        else
            min_diff=max(diff_dist);%异类最近样本
            r=min(same_dist(find(same_dist>min_diff)));%最小半径法,求最远同类点
        end
        output_temp.center=unlearn(k);
        output_temp.cdata=xk;
        output_temp.label=xk_label;
        output_temp.r=r;
        output_temp.cnt=1;
        output_temp.samples={unlearn(k)};
        train_learn(unlearn(k))=1;%覆盖中心标记为已学习
        for i=1:length(same_dist)
            if same_dist(i)>r
                if same_class(i)~=unlearn(k)
                    output_temp.samples=[output_temp.samples,same_class(i)];
                    train_learn(same_class(i))=1;%标记为已学习
                    output_temp.cnt=output_temp.cnt+1;
                end
            end
        end
        output=[output;output_temp];
        unlearn=find(train_learn==0);%更新未学习样本索引      
    end
    sum([output.cnt]);
    %%
    %testing
    %disp('start testing...');
    test_temp=[];result_label=[];test_dist=[];
    [l,w]=size(test_data);
    for t=1:l
        for j=1:length(output)
            test_dist(t,j)=sum(test_data(t,:).*output(j).cdata);%计算测试集样本与每个覆盖中心的内积
            if test_dist(t,j)>=output(j).r
                test_temp(t,j)=1;%test_temp逻辑值，=1表示在当前覆盖内
            else
                test_temp(t,j)=0;%=0表示不在当前覆盖内
            end
        end
    end
    test_temp(:,j+1)=sum(test_temp,2);%对test_temp求和
    for t=1:length(test_temp(:,j+1))
        if test_temp(t,j+1)==1%若sum(test_temp)=1，则表示该样本只落入一个覆盖内，则将其划分到此覆盖对应的类别；若!=1，则延迟决策
            index=find(test_temp(t,1:end-1)==1);
            result_label(t,1)=output(index).label;%result_label第一列表示延迟决策前的划分结果
            result_label(t,2)=output(index).label;%第二列表示延迟决策后的划分结果
            %延迟决策
        elseif test_temp(t,j+1)==0 %=0表示不属于任何一个覆盖，此时将样本划分到距离最近的覆盖中
            maxdist=max(test_dist(t,:));
            index=find(test_dist(t,:)==maxdist);
            result_label(t,1)=-1;%标记一下
            result_label(t,2)=output(index).label;
        else%表示落入多个划分中
            ind=find(test_temp(t,:)==1);
            maxdist=max(test_dist(t,ind));
            index=find(test_dist(t,:)==maxdist);
            result_label(t,1)=0;%标记一下
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
    %disp(['划分正确个数',num2str(right)]);
    %disp(['********正确率**********：',num2str(rate)]);
    rate_sum=rate+rate_sum;
end
    rate_ave=rate_sum/10;
    accuracy_ave(g,1)=rate_ave;
    disp(['rate_average per round:',num2str(rate_ave)]);
    rate_ave_sum=rate_ave_sum+rate_ave;
end
rate_ave_sum=rate_ave_sum/100;
disp(['**********************rate_average all rounds:',num2str(rate_ave_sum)]);
disp(['**********************方差为：',num2str(var(accuracy_ave))]);
