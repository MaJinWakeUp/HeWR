function xo = Weight_Heat(CNN, mean_CNN)
[~,~,K] = size(CNN);
S = sum(CNN,3);

CNN = reshape(CNN,[],K);
CNN = CNN'; 
CNN_org = CNN;
CNN = CNN - repmat(mean_CNN,1,size(CNN,2));
CNN = yael_vecs_normalize(CNN,2,0);

S0 = reshape(S,[],1);
A = CNN'*CNN;
ind = find(S0==0);
A (1:size(A,1)+1:size(A,1)^2) = 0;
A(ind,ind) = 0;
A(A<0.1) = 0;

constZ = 0.1;
Z =  constZ*mean(A(A>0)) ; % conductance to dummy ground

Fweights = get_potential_inv(A, Z);
Fweights = 1./Fweights; 
CNN = bsxfun(@times,CNN_org, Fweights);
xo = sum(CNN,2);
end