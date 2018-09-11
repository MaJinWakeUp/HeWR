function [ranks_QE, ranks_HR, ranks_QER] = cast_rerank(vecs, qvecs, ranks)
qnd_qe = 10;
qnd_hr = 800;

% QE
qvecs_qe = qvecs;
for i=1:size(qvecs,2)
       qvecs_qe(:,i) = mean([qvecs(:,i) vecs(:,ranks(1:qnd_qe,i))],2);  
end
qvecs_qe = yael_vecs_normalize(qvecs_qe,2,0);
ranks_QE = yael_nn(vecs, qvecs_qe, size(vecs,2), 'L2');

% Heat Rerank
ranks_HR = ranks;
for i=1:size(qvecs,2)
    temp = [vecs(:,ranks_HR(1:qnd_hr,i)) qvecs(:,i)];
    temp = temp - repmat(mean(temp,2),1,size(temp,2));
    A = temp'*temp;
    A (1:size(A,1)+1:size(A,1)^2) = 0;
    A(A<0) = 0;
    constZ = 0.1;
    Z =  constZ*mean(A(A>0)) ; % conductance to dummy ground
    reward = get_potential_inv_re(A, Z)';
    [~,ind] = sort(reward,'descend');
    temp =ranks_HR(1:qnd_hr,i);
    ranks_HR(1:qnd_hr,i) = temp(ind);
end

% QE + HeR
ranks_QER = ranks;
for i=1:size(qvecs,2)
    temp = [vecs(:,ranks_QER(1:qnd_hr,i)) qvecs_qe(:,i)];
    temp = temp - repmat(mean(temp,2),1,size(temp,2));
    A = temp'*temp;
    A (1:size(A,1)+1:size(A,1)^2) = 0;
    A(A<0) = 0;
    constZ = 0.1;
    Z =  constZ*mean(A(A>0)) ; % conductance to dummy ground
    reward = get_potential_inv_re(A, Z)';
    [~,ind] = sort(reward,'descend');
    temp =ranks_QER(1:qnd_hr,i);
    ranks_QER(1:qnd_hr,i) = temp(ind);
end
end

function reward = get_potential_inv_re(A, Z)

A = [A, Z*ones(size(A,1),1,'single');  ones(1,size(A,1),'single'), 0];
sA = sum(A,2);
A  = bsxfun(@rdivide, A, sA);

%Laplace
lap_mat = diag(sum(A,2)) - A ;
lap_mat = lap_mat(1:end-1,1:end-1);

%compute inverse of Laplace
inv_lap_mat = inv(lap_mat) ;

reward = inv_lap_mat(1:end-1,end);
end