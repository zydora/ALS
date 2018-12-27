function [TrainImages,TestImages,TrainLabels,TestLabels] = load_dataset(Dataset)
if strcmpi(Dataset,'MNIST')
    clear all;clc
    TrainImages = loadMNISTImages('train-images-idx3-ubyte');
    TestImages = loadMNISTImages('t10k-images-idx3-ubyte');
    TrainLabels = loadMNISTLabels('train-labels-idx1-ubyte');
    TestLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
    % SVHN
elseif strcmpi(Dataset,'SVHN')
    clear all;clc
    M = load('train_32x32.mat');
    N = load('test_32x32.mat');
    for i = 1:size(M.X,4)
        TrainImages(:,:,i) = rgb2gray(M.X(:,:,:,i));
    end
    for i = 1:size(N.X,4)
        TestImages(:,:,i) = rgb2gray(N.X(:,:,:,i));
    end
    TrainLabels = M.y-1;
    TestLabels = N.y-1;
    TrainImages = reshape(TrainImages,[32*32,size(TrainImages,3)]);
    TestImages = reshape(TestImages,[32*32,size(TestImages,3)]);
    TrainImages = im2double(TrainImages);
    TestImages = im2double(TestImages);
elseif strcmpi(Dataset,'CIFAR')
    clear all;clc
    untar('cifar-10-matlab.tar.gz','CIFAR');
    load_data;
    for i = 1:5
        TrainImages(i*10000-9999:i*10000,:) = M(i).data;
        TrainLabels(i*10000-9999:i*10000,:) = M(i).labels;
    end
    TrainImages = im2double(permute(TrainImages,[2,1]));
    TrainLabels = (permute(TrainLabels,[2,1]));
    TestImages = im2double(permute(N.data,[2,1]));
    TestLabels = (permute(N.labels,[2,1]));
elseif strcmpi(Dataset,'MSRC')

end
end