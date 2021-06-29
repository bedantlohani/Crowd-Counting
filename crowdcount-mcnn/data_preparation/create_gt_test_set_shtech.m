%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;

path = ['/home/bedant/crowdcount-mcnn/eval/'];
gt_path = ['/home/bedant/crowdcount-mcnn/output/ann.mat'];
gt_path_csv = ['/home/bedant/crowdcount-mcnn/output/gt/'];

pts = load(gt_path) ;
name = fieldnames(pts);

for i = 1:numel(name)
    input_img_name = strcat(path,name{i},'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    annPoints = pts.(name{i});
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);
    csvwrite([gt_path_csv, name{i}, '.csv'], im_density);
end
