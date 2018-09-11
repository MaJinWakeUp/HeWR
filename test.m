clear
addpath('./utils')
datatype = 'resnet';
dataset_test = 'oxford5k';
dataset_train = 'paris6k';

% the final dimension of vectors, default value is the original dimension
% dim = 512; 

path_traindata = ['./data/',datatype,'/',dataset_train,'/'];
path_testdata = ['./data/',datatype,'/',dataset_test,'/'];

% This step, feature extraction, may take a very long time.
% Or you can extract features by yourself and skip this step.
setup_features;

%% process training data to get whiten matrix
if exist([path_testdata,'vecs_train.mat'],'file') && exist([path_testdata,'mean_train.mat'],'file')
    load([path_testdata,'vecs_train.mat']);
    load([path_testdata,'mean_train.mat']);
else
    fprintf('Preprocessing training data...\n')
    conv3d_train = load([path_traindata,dataset_train,'_features']);
    
    % obtain mean value for later use
    mean_train = create_mean(conv3d_train.data);
    Path_save = [path_testdata,'mean_train.mat'];
    save(Path_save,'mean_train')
    
    % obtain training vectors for pca learning
    vecs_train = cellfun(@(x) Weight_Heat(x,mean_train),conv3d_train.data,'un',0);
    vecs_train = cell2mat(vecs_train);
    Path_save = [path_testdata,'vecs_train.mat'];
    save(Path_save,'vecs_train')
    clear conv3d_train;
    fprintf('end...\n')
end

fprintf('Learning PCA...\n')
vecs_train = preprocess(vecs_train);
[~, eigvec, eigval, Xm] = yael_pca (vecs_train);
fprintf('end...\n')

%% process test data
if exist([path_testdata,'vecs_test.mat'],'file')
    load([path_testdata,'vecs_test.mat']);
else
    fprintf('Preprocessing test data...\n')
    conv3d_test = load([path_testdata,dataset_test,'_features']);
    vecs_test = cellfun(@(x) Weight_Heat(x,mean_train),conv3d_test.data,'un',0);
    vecs_test = cell2mat(vecs_test);
    Path_save = [path_testdata,'vecs_test.mat'];
    save(Path_save,'vecs_test')
    clear conv3d_test;
    fprintf('end...\n')
end
fprintf('Applying PCA on test dataset...\n')
vecs_test = preprocess(vecs_test);
if ~exist('dim','var')
    dim = size(vecs_test,1);
end
vecs_test = apply_whiten (vecs_test, Xm, eigvec, eigval, dim);
vecs_test = yael_vecs_normalize(vecs_test,2,0);
fprintf('end...\n')

%% process query images
switch dataset_test
    case {'oxford5k','paris6k'}
        if exist([path_testdata,'vecs_query.mat'],'file')
            load([path_testdata,'vecs_query.mat']);
        else
            fprintf('Processing query images...\n');
            load([path_testdata,'query_features.mat']);
            qvecs = cellfun(@(x) Weight_Heat(x,mean_train),qim,'un',0);
            qvecs = cell2mat(qvecs);
            Path_save = [path_testdata,'vecs_query.mat'];
            save(Path_save,'qvecs')
            clear qim;
            fprintf('end...\n');
        end
        fprintf('Applying PCA on query dataset...\n')
        qvecs = preprocess(qvecs);
        qvecs = apply_whiten (qvecs, Xm, eigvec, eigval, dim);
        qvecs = yael_vecs_normalize(qvecs,2,0);
        load (['./data/',datatype,'/gnd_',dataset_test])
        fprintf('end...\n');
        
    case 'holidays'
        load (['./data/',datatype,'/gnd_',dataset_test])
        qvecs = vecs_test(:,qidx);
end

%% -------------------image search--------------------
[ranks,sim] = yael_nn(vecs_test, qvecs, size(vecs_test,2), 'L2');
[map,aps] = compute_map (ranks, gnd);
fprintf('%s  map, without rerank = %.4f\n',dataset_test,map);

rerank = 1;
if rerank
    [ranks_QE, ranks_HeR, ranks_QER] = cast_rerank(vecs_test, qvecs, ranks);
    [map_qe,~] = compute_map (ranks_QE, gnd);
    [map_he,~] = compute_map (ranks_HeR, gnd);
    [map_qer,~] = compute_map (ranks_QER, gnd);
    fprintf('map, after qe = %.4f, after HeR = %.4f, after QE+HeR = %.4f.\n',...
        map_qe, map_he, map_qer);
end
