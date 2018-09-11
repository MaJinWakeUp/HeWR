function reward = get_potential_inv(A, Z)

A = [A, Z*ones(size(A,1),1,'single');  ones(1,size(A,1),'single'), 0];
sA = sum(A,2);
A  = bsxfun(@rdivide, A, sA);

% Laplace
lap_mat = diag(sum(A,2)) - A ;
lap_mat = lap_mat(1:end-1,1:end-1);

% compute inverse of Laplace
inv_lap_mat = inv(lap_mat) ;

diag_ind = 1:size(lap_mat,1)+1:size(lap_mat,1)^2;
deno = inv_lap_mat(diag_ind);
inv_lap_mat(diag_ind) = 0;
reward = sum(inv_lap_mat)./deno;
reward = (reward)/size(A,1);
end