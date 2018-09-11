% setup_features
% this step checks either you have extracted features or not
% if yes, it continues
% if not, this script is supposed to extract features to specific location
if ~exist([path_traindata,dataset_train,'_features.mat'],'file')
    fprintf('No train data features detected. Starting extract features...\n')
    data = feature_extract(datatype,dataset_train);
    Path_save = [path_traindata,dataset_train,'_features.mat'];
    save(Path_save,'data')
    clear data
end

if ~exist([path_testdata,dataset_test,'_features.mat'],'file')
    fprintf('No test data features detected. Starting extract features...\n')
    data = feature_extract(datatype,dataset_test);
    Path_save = [path_testdata,dataset_test,'_features.mat'];
    save(Path_save,'data')
    clear data
end

switch dataset_test
    case {'oxford5k','paris6k'}
        if ~exist([path_testdata,'query_features.mat'],'file')
            fprintf('No query data features detected. Starting extract query features...\n')
            data = feature_query(datatype,dataset_test);
            Path_save = [path_testdata,'query_features.mat'];
            save(Path_save,'data')
            clear data         
        end
end