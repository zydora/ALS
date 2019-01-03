for j=1:5 %读取训练集数据
    M(j) = load(['data_batch_' num2str(j) '.mat']);
    
end
N = load('test_batch.mat');