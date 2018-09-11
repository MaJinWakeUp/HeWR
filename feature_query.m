% feature_query, extract query features for later use
% this function only works for vgg16 and resnet
% for SiaMAC, please refer to its original paper
function data = feature_query(datatype,dataset)
cd ../matconvnet/matlab
vl_setupnn;
cd ../../HeWR
addpath('./utils');

% change this path to where you store your dataset images
% in our computer, all dataset images are contained in
% D:/imagesearch/
im_folder = ['D:/imagesearch/', dataset, '/'];
load(['./data/',datatype,'/gnd_', dataset, '.mat']);    

switch datatype
    case 'vgg16'
        modelfn = 'imagenet-vgg-verydeep-16.mat';  lid = 31;		% use VGG
        net = load(['../matconvnet/' modelfn]);
        net.layers = {net.layers{1:lid}}; % remove fully connected layers

        qimlist = {imlist{qidx}};
        qim = arrayfun(@(x) crop_qim([im_folder, qimlist{x}, '.jpg'], gnd(x).bbx), 1:numel(qidx), 'un', 0);
        for imnum = 1:numel(qim)
            tic
            im = single(qim{imnum}) ;
            for i=1:3
                im_(:,:,i) = im(:,:,i) - mean(net.meta.normalization.averageImage(i));
            end
            rnet = vl_simplenn(net, im_);  
            data{imnum} = max(rnet(end).x, 0);
            clear im_
            toc
        end
        
    case 'resnet'
        net = dagnn.DagNN.loadobj(load('../matconvnet/imagenet-resnet-50-dag.mat')) ;
        net.mode = 'test' ;
        net.vars(173).precious = true;
        
        qimlist = {imlist{qidx}};
        qim = arrayfun(@(x) crop_qim([im_folder, qimlist{x}, '.jpg'], gnd(x).bbx), 1:numel(qidx), 'un', 0);
        minsize = 224;
        mean_value = mean(mean(net.meta.normalization.averageImage));
        for imnum = 1:numel(qim)
            tic
            im = single(qim{imnum}) ;
            for i=1:3
                im_(:,:,i) = im(:,:,i) - mean_value(i);
            end
            if min(size(im_, 1), size(im_, 2)) < minsize
                im_ = pad2minsize(im_, minsize, 0);
            end
            net.eval({'data', im_});
            data{imnum} = net.vars(173).value;

            clear im_
            toc
        end
end
end
