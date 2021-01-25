addpath 'util'
clear all; clc; close all; % clean up!

trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' }; % Note, we use the same IDs as in the Nature Machine Intelligence Paper.
testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};
allsites = cat(2, trainingsites, testsites );

% folders for efficientdet:
edet_base_folder = './Yet-Another-EfficientDet-Pytorch/datasets/cv_project/';
val_img_folder = './eval/';
train_img_folder = './train/';
annotations_folder = './annotations/';

% create directories
mkdir(fullfile(edet_base_folder,train_img_folder));
mkdir(fullfile(edet_base_folder,val_img_folder));
mkdir(fullfile(edet_base_folder,annotations_folder));

% read in json files and convert it to structs
[training_struct, test_struct] = initialize_json();

% id's for efficientdet json file
training_ann_id = 0;
training_img_id = 0;
test_ann_id = 0;
test_img_id = 0;

% folder where modified labels created with 'relabel_data.m' are saved, if labels are modified
modified_base_folder = './modified_labels/'
for i_site = 1:length(allsites)
    for z_offset = [-4, -2, 0, 2, 4]
        site = allsites{i_site};
    
        datapath = fullfile( './data/', site ); 
        datapath_labels = datapath;
        
        % if modified labels were created, use those for preprocessing instead of original labels
        if exist(modified_base_folder, 'dir')
            datapath_labels = fullfile('./modified_labels/', site);
        end
        
        if ~isfolder(fullfile( datapath ))
           error( 'folder %s does not exist. Did you download additional data?', datapath );
        end

        % don't augment z-parameter for test images
        if contains(site, "T") && z_offset~=0 
            continue
        end

        fprintf("\nSite: %s, z-offset: %d\n", site, z_offset);
       
        % create folder in which to save images and bb for current z-parameter-offset
        if contains(site, "F")
            resultsfolder = fullfile( './results/', sprintf("%s_offset%d", site, z_offset));
        else
            resultsfolder = fullfile( './results/', site);
        end
        mkdir(resultsfolder);
        
        %%%%%%%%
        % image integration
        %%%%%%%%

        thermalParams = load( './data/camParams_thermal.mat' );
        % Note: line numbers might not be consecutive and they don't start at index
        % 1. So we loop over the posibilities:
        for linenumber = 1:99
            if ~isfile(fullfile( datapath, '/Poses/', [num2str(linenumber) '.json'] ))
                continue % SKIP!
            end
        
            json = readJSON( fullfile( datapath, '/Poses/', [num2str(linenumber) '.json'] ) );
            images = json.images; clear json;
            
            try
               json = readJSON( fullfile( datapath_labels, '/Labels/', ['Label' num2str(linenumber) '.json'] ) );
               labels = json.Labels; clear json;
            catch
               warning( 'no Labels defined!!!' ); 
               labels = []; % empty
            end
        
            K = thermalParams.cameraParams.IntrinsicMatrix; % intrinsic matrix, is the same for all images
            Ms = {};
        
            thermalpath = fullfile( datapath, 'Images', num2str(linenumber) );
        
            for i_label = 1:length(images)
               thermal = undistortImage( imread(fullfile(thermalpath,images(i_label).imagefile)), ...
                   thermalParams.cameraParams );
               M = images(i_label).M3x4;
               M(4,:) = [0,0,0,1];
               Ms{i_label} = M;
        
            end
        
            refId = (round(length(images)/2))+1; % compute center by taking the average id!
            imgr = undistortImage( imread(fullfile(thermalpath,images(refId).imagefile)), ...
                   thermalParams.cameraParams );
            M1 = Ms{refId};
            R1 = M1(1:3,1:3)';
            t1 = M1(1:3,4)';
            range = [min(imgr(:)), max(imgr(:))];
            integral = zeros(size(imgr),'double');
            count = zeros(size(imgr),'double');
        
            for i_label = 1:length(images)
                original = imread(fullfile(thermalpath,images(i_label).imagefile));
                % uncomment the following lines to perform contrast enhancement on single images before image integration
                %original_uint8 = uint8(255*mat2gray(original)); % convert to uint8 for adapthisteq to work
                %original_CLAHE = adapthisteq(original_uint8, 'NumTiles', [8,8], 'ClipLimit', 0.01, 'NBins', 256, 'Range', 'full', 'Distribution', 'uniform');
                %original = im2uint16(original_CLAHE); % convert back to uint16
                img2 = undistortImage( original, ...
                       thermalParams.cameraParams );
        
        
                M2 = Ms{i_label};
                R2 = M2(1:3,1:3)';
                t2 = M2(1:3,4)';
        
                % relative 
                R = R1' * R2;
                t = t2 - t1 * R;
        
                z = getAGL( site ) + z_offset; % meters
                % the checkerboard is ~900 millimeters away
                % the tree in the background is ~100000 millimeters (100 m)
                P = (inv(K) * R * K ); 
                P_transl =  (t * K);
                P_ = P; % copy
                P_(3,:) = P_(3,:) + P_transl./z; % add translation
                tform = projective2d( P_ );
        
                % --- warp images ---
                % warp onto reference image
                warped2 = double(imwarp(img2,tform.invert(), 'OutputView',imref2d(size(imgr))));
                warped2(warped2==0) = NaN; % border introduced by imwarp are replaced by nan
        
                count(~isnan(warped2)) = count(~isnan(warped2)) + 1;
                integral(~isnan(warped2)) = integral(~isnan(warped2)) + warped2(~isnan(warped2));
        
            end
            lfr = integral ./ count;
        
            % project labels
            K_ = K; K_(4,4) = 1.0; % make sure intrinsic is 4x4
            
            % STORE
            % normalize to [0 1]
            img = lfr - min(lfr(:));
            img = img ./ max(img(:));
            % increase contrast using CLAHE
            %img = adapthisteq(img)
    
            if site(1) == 'F'
                % save in results folder
                imwrite( img, fullfile(resultsfolder, sprintf('%s_line%d_offset%d.png', site, linenumber, z_offset)));
                % save for efficientdet
                imwrite(img, fullfile(edet_base_folder, train_img_folder, sprintf('%s_line%d_offset%d.png', site, linenumber, z_offset)));
                training_struct.images = [training_struct.images, struct('id', training_img_id, ...
                    'file_name', sprintf('%s_line%d_offset%d.png', site, linenumber, z_offset), ...
                    'width', size(img, 2), 'height', size(img, 1), 'date_captured', '', 'license', 1, ...
                    'coco_url', '', 'flickr_url', '')]; 
            elseif site(1) == 'T'
                imwrite(img, fullfile(resultsfolder, sprintf('%s_line%d.png', site, linenumber)));
                imwrite(img, fullfile(edet_base_folder, val_img_folder, sprintf('%s_line%d.png', site, linenumber)));
                test_struct.images = [test_struct.images, struct('id', test_img_id, ...
                    'file_name', sprintf('%s_line%d.png', site, linenumber), ...
                    'width', size(img, 2), 'height', size(img, 1), 'date_captured', '', 'license', 1, ...
                    'coco_url', '', 'flickr_url', '')];
            else
                warning('site not recognized')
            end
    
            % store AABBs
            if ~isempty(labels) && ~isempty({labels.poly})
                % read bb
                bbs = saveLabels( {labels.poly}, size(integral), [] );
                % make sure bounding boxes don't consist of just zeros
                % if all coordinates of a bb are smaller (or equ. to) 1, remove it
                bbs(all(bbs<=1,2),:)=[]; 
    
                % check if bb-matrix is still non-empty
                if size(bbs,1) > 0
                    % define bounding box matrices according to matlab indexing
                    x_left = bbs(:,1);
                    x_right = bbs(:,2);
                    y_bottom = bbs(:,3);
                    y_top = bbs(:,4);
                    bb_matrix = [x_left, y_bottom, x_right - x_left, y_top - y_bottom];
    
                    % define bounding box parameters according to python indexing as required for efficientdet 
                    % (i.e. subtract 1 for compatibility with python indexing)
                    x_left = round(x_left - 1);
                    x_right = round(x_right - 1);
                    y_bottom = round(y_bottom - 1);
                    y_top = round(y_top - 1);
                    width = x_right - x_left;
                    height = y_top - y_bottom;
    
                    % save each bounding box as csv file in results folder as well as in
                    % struct which will later be saved as .json for efficientdet
                    for i_bb = 1:size(x_left, 1)
                        % check if current image/bb belongs to training or test site
                        if site(1) == 'F'
                            writematrix(bb_matrix, fullfile(resultsfolder, sprintf('%s_line%d_offset%d.csv', site, linenumber, z_offset)));
                            % add entry in struct with necessary fields for efficientdet
                            training_struct.annotations = [training_struct.annotations, struct(...
                                'id', training_ann_id, 'image_id', training_img_id, 'category_id', 1, ...
                                'iscrowd', 0, 'area', height(i_bb)*width(i_bb), ...
                                'bbox', [x_left(i_bb), y_bottom(i_bb), width(i_bb), height(i_bb)],...
                                'segmentation', {{[x_left(i_bb), y_bottom(i_bb), x_right(i_bb), ...
                                y_bottom(i_bb), x_right(i_bb), y_top(i_bb), x_left(i_bb), y_top(i_bb)]}})];
                            training_ann_id = training_ann_id + 1;
                        elseif site(1) == 'T'
                            writematrix(bb_matrix, fullfile(resultsfolder, sprintf( '%s_line%d.csv', site, linenumber)));
                            test_struct.annotations = [test_struct.annotations, struct(...
                                'id', test_ann_id, 'image_id', test_img_id, 'category_id', 1, ...
                                'iscrowd', 0, 'area', height(i_bb)*width(i_bb), ...
                                'bbox', [x_left(i_bb), y_bottom(i_bb), width(i_bb), height(i_bb)],...
                                'segmentation', {{[x_left(i_bb), y_bottom(i_bb), x_right(i_bb), ...
                                y_bottom(i_bb), x_right(i_bb), y_top(i_bb), x_left(i_bb), y_top(i_bb)]}})];
                            test_ann_id = test_ann_id + 1;
                        else
                            warning('site not recognized')
                        end
                    end
                else % if bb matrix was empty after removing 0-size bb, simply save a zero-vector in csv-file
                    if site(1) == 'F'
                        writematrix([0 0 0 0], fullfile(resultsfolder, sprintf('%s_line%d_offset%d.csv', site, linenumber, z_offset)));
                    else
                        writematrix([0 0 0 0], fullfile(resultsfolder, sprintf( '%s_line%d.csv', site, linenumber)));
                    end
                end
            else % if already 'labels' or '{labels.poly}' was empty, also save 0-vector
                if site(1) == 'F'
                    writematrix([0 0 0 0], fullfile(resultsfolder, sprintf('%s_line%d_offset%d.csv', site, linenumber, z_offset)));
                else
                    writematrix([0 0 0 0], fullfile(resultsfolder, sprintf( '%s_line%d.csv', site, linenumber)));
                end
            end
    
            % increase corresponding image id by 1
            if site(1) == 'F'
                training_img_id = training_img_id + 1;
            elseif site(1) == 'T'
                test_img_id = test_img_id + 1;
            end
        end
    end
end

% data augmentation
training_struct
training_struct = data_augmentation(training_struct);

% encode for saving as json
training_json_str = jsonencode(training_struct);
test_json_str = jsonencode(test_struct);

% save as json files
fid = fopen(fullfile(edet_base_folder, annotations_folder, 'instances_train.json'), 'w+');
fwrite(fid, training_json_str, 'char');
fclose(fid);

fid = fopen(fullfile(edet_base_folder, annotations_folder, 'instances_eval.json'), 'w+');
fwrite(fid, test_json_str, 'char');
fclose(fid);

