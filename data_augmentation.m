img_folder = './results/';
trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' }; 
testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};

num_imgs = containers.Map() 
%% Create dataset with training images
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


% there are a lot of possibilites for data augmentation using the function
% randomAffine2d(), e.g. cropping, translation, rotation, reflection, shearing,...
% see https://de.mathworks.com/help/deeplearning/ug/bounding-box-augmentation-using-computer-vision-toolbox.html

%% data augmentation
imds = imageDatastore(table2cell(training_img(:,1)))
blds = boxLabelDatastore(training_img(:,2))
trainingData = combine(imds,blds)

augmentedTrainingData = transform(trainingData,...
    @(data)augmentationFunction(data, "XReflection",true, "YReflection",true,...
    "Rotation",[-180 180], "Scale",[1 1], "XShear",[0 0], "YShear",[0 0],...
    "XTranslation",[0 0], "YTranslation",[0 0]));
% Read all the augmented data.
data = readall(augmentedTrainingData);

length(data)
size(training_img,1)
size(data)


% Display the augmented image and box label data.
numObservations=20
start_index = 1
collage = cell(numObservations,1);
for k = 1:(numObservations)
    I = data{k+start_index,1};
    bbox = data{k+start_index,2};
    labels = data{k+start_index,3};
    collage{k} = insertObjectAnnotation(I,'rectangle',bbox,labels, ...
        'LineWidth',1,'FontSize',10);
end
montage(collage)



% create folder to save augmented images
save_folder = fullfile(img_folder, "augmented_images"); 
if ~isdir(save_folder)
    mkdir(save_folder);
end

% save an augmented image and its bb
i = 1
img = data{i,1};
imwrite(img, fullfile(save_folder, sprintf('img_%d.png', i)));
writematrix(data{i,2}, fullfile(save_folder, sprintf('img_%d.csv', i)));


% read an augmented image and its bb
img = imread(fullfile(save_folder, sprintf('img_%d.png', i)));
bb = csvread(fullfile(save_folder, sprintf('img_%d.csv', i)));
bb = bb(any(bb ~= 0,2),:);

annotation = 'bb';
for i=1:size(bb, 1)
    img = insertObjectAnnotation(img,'rectangle',bb(i,:),annotation);
end
imshow(img)


