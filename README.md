# Introduction
This is a Matlab implementation of our paper:

**Deep Feature Aggregation and Image Re-ranking with Heat Diffusion for Image Retrieval**

In this repository, we show the image search procedure of our method, HeWR, on three dataset: Holidays, Oxford5k, Paris6k.
Note: Feature extraction with siaMac is not provided yet, we will add it in later edition.
***
# Prerequisites
There are something you need before you run this program.
## Dependence
* [Matconvnet][1]. This is a MATLAB toolbox implementing CNNs for computer vision applications. We use this tool to extract features of different models.
* Optional but recommended: [Library yael][2]. Yael is a library implementing computationally intensive functions used in large scale image retrieval. ( Functions needed in this experiment are already contained in folder `utils`. ) 
## Dataset
* [Oxford5k][3] consists of 5062 images collected from Flickr by searching for particular Oxford landmarks.
* [Paris6k][4] consists of 6412 images collected from Flickr by searching for particular Paris landmarks. In our experiments, we delete the 20 corrupted images and use the other **6392** images.
* [Inria Holidays][5] is a set of images which mainly contains some holidays photos.
***
# How to run experiment
1. Download and install [Matconvnet][1].
    1. Download the pre-trained [vgg-16 model][6] and [resnet model][7] to matconvnet root dir.
2. Download dataset images.
3. Download and unzip this repository, the **HeWR** folder and **matconvnet** folder should in a same folder.
4. Run experiment, the `test.m` is supposed to do the whole procedure. (You may need to change the variable *im_folder* in `feature_extract.m` and `feature_query.m` to location where you store dataset images.)

***
**If you have any question, please contact:**  
*Jin Ma, m799133891@stu.xjtu.edu.cn* or *Shanmin Pang, pangsm@xjtu.edu.cn*

[1]: http://www.vlfeat.org/matconvnet/ "matconvnet home"
[2]: https://gforge.inria.fr/projects/yael/ "yael home"
[3]: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ "Oxford dataset"
[4]: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/ "Paris dataset"
[5]: http://lear.inrialpes.fr/~jegou/data.php#holidays "Holidays dataset"
[6]: http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat "vgg-16 model"
[7]: http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat "resnet model"
