%# %%
addpath 'util'
img_folder = './results/';
trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11'}; 
testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};
allsites = cat(2, trainingsites, testsites );

modified_base_folder = './modified_labels/'
if ~exist(modified_base_folder, 'dir')
    sprintf('Copying labels from ./data to %s', modified_base_folder)
    for i=1:size(allsites,2)
        site = allsites{i};
        new_folder = fullfile(modified_base_folder, site, 'Labels');
        mkdir(new_folder);

        source_folder = fullfile('./data/', site, 'Labels');
        sprintf('copying from %s to %s', source_folder, new_folder);
        copyfile(fullfile(source_folder,'/*'), new_folder);
    end
end

num_imgs = containers.Map() 
%% Create dataset with training images
training_img = cell2table(cell(0,3), "VariableNames", ["path","AABB","json_path"]);
for i = 1:length(trainingsites)
    site = trainingsites{i};
    site_imgs = dir(fullfile(img_folder, site, '/*.png'));
    site_imgs = {site_imgs.name};
    site_imgs = natsortfiles(site_imgs);
    num_imgs(trainingsites{i}) = length(site_imgs);
    for j = 1:length(site_imgs)
        name = replace(site_imgs{j}, '.png', '');

        % image name
        img_path = fullfile(img_folder, site, sprintf('%s.png', name));

        % bounding box
        try
            bb = csvread(fullfile(img_folder, site, sprintf('%s.csv', name)));
        catch ME
            fprintf(ME.message);
            fprintf(" File: %s\n", ...
                fullfile(img_folder, site, sprintf('%s.csv', name))); 
        end

        % remove rows with only zeros
        bb = bb(any(bb ~= 0,2),:);
        % add to table
        json_path = fullfile(modified_base_folder, site, '/Labels/', sprintf('Label%d.json', j));
        training_img = [training_img; {img_path, bb, json_path}];
    end
end

relabeling_func(training_img);

%# %%
