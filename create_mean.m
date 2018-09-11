% create mean_CNN 
function mean_value = create_mean(data)
nfeats = 0;
dim = size(data{1},3);
mean_value = zeros(dim,1,'single');

for i = 1:numel(data)
    CNN = data{i};
    CNN = reshape(CNN,[],dim);
    CNN = CNN'; % 每一列是一个描述子；
    
    mean_value = mean_value + sum(CNN,2);
    nfeats = nfeats + size(CNN,2);
end

mean_value = mean_value/nfeats;
end
