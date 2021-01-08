function training_struct = data_augmentation(training_struct)
    img_folder = './results/';

    % Get a list of all folders in this img_folder.
    files = dir(img_folder);
    folders = files([files.isdir]);
    trainingsites = {};
    for fname = {folders.name}
        if contains(fname, "F")
            trainingsites = [trainingsites, fname];
        end
    end

    %% Create dataset with training images
    training_img = cell2table(cell(0,2), "VariableNames", ["path","AABB"]);
    for i = 1:length(trainingsites)
        site = trainingsites{i};
        site_imgs = dir(fullfile(img_folder, site, '/*.png'));
        %fprintf("\nSite: %s - %d images", site, size(site_imgs,1))
        for j = 1:length(site_imgs)
            name = replace(site_imgs(j).name, '.png', '');
    
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
            training_img = [training_img; {img_path, bb}];
        end
    end

    %% store image data in datastore 
    imds = imageDatastore(table2cell(training_img(:,1)));
    blds = boxLabelDatastore(training_img(:,2));
    trainingData = combine(imds,blds);
    
    %% create folder to save augmented images
    save_folder = fullfile(img_folder, "augmented_images"); 
    if ~isdir(save_folder)
        mkdir(save_folder);
    end
    

    %% -----------------------------------------------------------
    % This is the relevant part for defining augmentation operations
    %
    % Define different augmentation operations 
    % data augmentation is done with the function randomAffine2D() inside
    % of augmentation_function()
    % default arguments for randomAffine2D():
    % ["XReflection",false, "YReflection",false, "Rotation",[0 0], 
    % "Scale",[1 1], "XShear",[0 0], "YShear",[0 0], 
    % "XTranslation",[0 0], "YTranslation",[0 0]]
    %
    % notes on some of the arguments:
    % - XShear and YShear values must be between -90 and 90
    % - XTranslation and YTranslation is given in number of pixels
    % - Scale is probably not useful for our task, since our images always 
    %   have the same resolution anyways
    
    % define anonymous function g which creates another anonymous function, 
    % which chooses a value in the given in the given interval according to 
    % a random distribution with standard deviation of diff(interval)/6 
    % -> makes it so that ~68% of the randomely picked values will be in the 
    % middle 33.3% of the interval, ~95% in the middle 66.6%, and no 
    % values outside the interval.
    g = @(val_min, val_max) (@() max(val_min, min(val_max, ...
        randn*(val_max-val_min)/6 + (val_max+val_min)/2)));

    xyrefl = {'XReflection', true, 'YReflection', true, ...
        'Rotation', g(-180,180), 'Scale', g(0.5,1.5), ...
        'XShear', g(-40, 40), 'YShear', g(-40,40),...
        'XTranslation', g(-150,150), 'YTranslation', g(-150,150)};
    xrefl = {'XReflection', true, 'YReflection', false, ...
        'Rotation', g(-180,180), 'Scale', g(0.5,1.5), ...
        'XShear', g(-40, 40), 'YShear', g(-40,40),...
        'XTranslation', g(-150,150), 'YTranslation', g(-150,150)};
    yrefl = {'XReflection', false, 'YReflection', true, ...
        'Rotation', g(-180,180), 'Scale', g(0.5,1.5), ...
        'XShear', g(-40, 40), 'YShear', g(-40,40),...
        'XTranslation', g(-150,150), 'YTranslation', g(-150,150)};
    norefl = {'XReflection', false, 'YReflection', false, ...
        'Rotation', g(-180,180), 'Scale', g(0.5,1.5), ...
        'XShear', g(-40, 40), 'YShear', g(-40,40),...
        'XTranslation', g(-150,150), 'YTranslation', g(-150,150)};

    augmentations = [xyrefl; xrefl; yrefl; norefl]

    %% get data to augment 
    edet_base_folder = './Yet-Another-EfficientDet-Pytorch/datasets/cv_project/';
    train_img_folder = './train/';
    annotations_folder = './annotations/';
    
    % get latest image and annotations ids
    img_id = training_struct.images(end).id;
    ann_id = training_struct.annotations(end).id;

    %% loop over the different data augmentations the specified number of times and save
    %% resulting images and their bb
    num_per_refl = 5
    fprintf('After the following loop you will have %d new images',...
        size(augmentations,1)*size(training_img,1)*num_per_refl)

    for n = 1:num_per_refl
        fprintf("\nloop iteration %d out %d...", n, num_per_refl)
        for i = 1:size(augmentations,1)
            augmentedTrainingData = transform(trainingData,...
                @(data)augmentation_function(data, augmentations{i,:}));

            % Read all the augmented data.
            augmented_data = readall(augmentedTrainingData);
            num_imgs = size(augmented_data,1);
        
            % save augmented images and their bb 
            for j = 1:num_imgs
                img = augmented_data{j,1};
                imwrite(img, fullfile(save_folder, sprintf('img_%d.png', ...
                    (i-1)*num_imgs+j)));
                imwrite(img, fullfile(edet_base_folder, train_img_folder, sprintf('img_%d.png', img_id)));
                training_struct.images = [training_struct.images, struct('id', img_id, ...
                    'file_name', sprintf('img_%d.png', img_id), ...
                    'width', size(img, 2), 'height', size(img, 1), 'date_captured', '', 'license', 1, ...
                    'coco_url', '', 'flickr_url', '')]; 
        
                if size(augmented_data{j,2}, 1) > 0
                    % save for acf trainer
                    writematrix(augmented_data{j,2}, fullfile(save_folder, ...
                        sprintf('img_%d.csv', (i-1)*num_imgs+j)));
        
                    % append to training data for efficientdet
                    x_left = augmented_data{j,2}(:,1) - 1;
                    y_bottom = augmented_data{j,2}(:,2) - 1;
                    width = augmented_data{j,2}(:,3);
                    height = augmented_data{j,2}(:,4);
                    x_right = x_left + width;
                    y_top = y_bottom + height;
                    for i_bb = 1:size(x_left, 1)
                        training_struct.annotations = [training_struct.annotations, struct(...
                            'id', ann_id, 'image_id', img_id, 'category_id', 1, ...
                            'iscrowd', 0, 'area', height(i_bb)*width(i_bb), ...
                            'bbox', [x_left(i_bb), y_bottom(i_bb), width(i_bb), height(i_bb)],...
                            'segmentation', {{[x_left(i_bb), y_bottom(i_bb), x_right(i_bb), ...
                            y_bottom(i_bb), x_right(i_bb), y_top(i_bb), x_left(i_bb), y_top(i_bb)]}})];
                        ann_id = ann_id + 1;
                    end
                else
                    % save for acf trainer
                    writematrix([0 0 0 0], fullfile(save_folder, ...
                        sprintf('img_%d.csv', (i-1)*num_imgs+j)));
        
                end
                img_id = img_id + 1;
            end
        end
    end
end

