function training_struct = data_augmentation(training_struct)

    addpath 'util'
    img_folder = './results/';
    trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' }; 
    testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};
    
    %% Create dataset with training images
    
    num_imgs = containers.Map() 
    training_img = cell2table(cell(0,2), "VariableNames", ["path","AABB"]);
    for i = 1:length(trainingsites)
        site = trainingsites{i};
        site_imgs = dir(fullfile(img_folder, site, '/*.png'));
        num_imgs(trainingsites{i}) = length(site_imgs);
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
    
    keys(num_imgs)
    values(num_imgs)
    sum(cell2mat(values(num_imgs))) % total number of images that SHOULD be present
    size(training_img, 1) % number of images that ARE actually present
    head(training_img)
    
    %% store image data in datastore 
    imds = imageDatastore(table2cell(training_img(:,1)))
    blds = boxLabelDatastore(training_img(:,2))
    trainingData = combine(imds,blds)
    
    %% create folder to save augmented images
    save_folder = fullfile(img_folder, "augmented_images"); 
    if ~isdir(save_folder)
        mkdir(save_folder);
    end
    
    %% Define different augmentation operations 
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
    
    % create map with default values for the possible arguments
    argument_keys = {'XReflection', 'YReflection', 'Rotation', 'Scale', ...
        'XShear', 'YShear', 'XTranslation', 'YTranslation'};
    argument_vals = {false, false, [0 0], [1 1], [0 0], [0 0], [0 0], [0 0]};
    default_args = containers.Map(argument_keys, argument_vals);
    
    % create array with different rows corresponding to arguments
    % for different augmentation operations
    args = {}; % initialize it with empty array
    
    % add data augmentation with x-reflection
    new_args = containers.Map(default_args.keys, default_args.values);
    new_args('XReflection') = true; % set XReflection argument to true
    args = add_arguments(new_args, args); % add as a row to args array to be looped over later
    
    % add data augmentation with y-reflection
    new_args = containers.Map(default_args.keys, default_args.values);
    new_args('YReflection') = true; % now set YReflection to true
    args = add_arguments(new_args, args); % add to args array
    
    % use reflection on x and y
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('YReflection') = true; 
    new_args('XReflection') = true;
    args = add_arguments(new_args, args); % add to args array
    
    % x reflection and rotation between -45 and 45 degrees
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('XReflection') = true; 
    new_args('Rotation') = [-45, 45]; 
    args = add_arguments(new_args, args); % add to args array
    
    % x and y reflection and rotation between 0 and 180 degrees
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('XReflection') = true; 
    new_args('YReflection') = true; 
    new_args('Rotation') = [0, 180]; 
    args = add_arguments(new_args, args); % add to args array
    
    % y reflection, rotation, xshear
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('YReflection') = true; 
    new_args('Rotation') = [-20, 10]; 
    new_args('XShear') = [-45, 45]; 
    args = add_arguments(new_args, args); % add to args array
    
    % only x and y shear
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('XShear') = [-20, 20]; 
    new_args('YShear') = [-45, 45]; 
    args = add_arguments(new_args, args); % add to args array
    
    % x and y shear with reflections
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('XReflection') = true; 
    new_args('YReflection') = true; 
    new_args('XShear') = [-20, 20]; 
    new_args('YShear') = [-45, 45]; 
    args = add_arguments(new_args, args); % add to args array
    
    % rotation, x translation
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('Rotation') = [-20, 20]; 
    new_args('XTranslation') = [-100, 100]; 
    args = add_arguments(new_args, args); % add to args array
    
    % x and y reflection, x and y translation, y shear
    new_args = containers.Map(default_args.keys, default_args.values); 
    new_args('XReflection') = true; 
    new_args('YReflection') = true; 
    new_args('XTranslation') = [-30, 50]; 
    new_args('YTranslation') = [-80, 80]; 
    new_args('YShear') = [-45, 45]; 
    args = add_arguments(new_args, args); % add to args array
    
    %% and many more combinations could be done...
    fprintf('After the following loop you will have %d new images',...
        size(args,1)*size(training_img,1))
    
    %% get data to augment 
    edet_base_folder = './Yet-Another-EfficientDet-Pytorch/datasets/cv_project/';
    train_img_folder = './train/';
    annotations_folder = './annotations/';
    
    % get latest image and annotations ids
    img_id = training_struct.images(end).id;
    ann_id = training_struct.annotations(end).id;
    
    %% loop over all the different data augmentations and save
    %% resulting images and their bb
    for i = 1:size(args, 1)
        fprintf("\nrow %d of argument table...", i)
        augmentedTrainingData = transform(trainingData,...
            @(data)augmentation_function(data, args{i,:}));
    
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
    
                % TODO: Do we also have to save empty bb annotations for efficientdet? if so, in what format?
            end
            img_id = img_id + 1;
        end
    end
end

