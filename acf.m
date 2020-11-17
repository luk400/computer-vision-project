img_folder = './results/';
trainingsites = { 'F0', 'F1', 'F2', 'F3', 'F5', 'F6', 'F8', 'F9', 'F10', 'F11' }; 
testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};

%% Create dataset with training images
training_img = cell2table(cell(0,2), "VariableNames", ["path","AABB"]);
for i = 1:length(trainingsites)
    site = trainingsites{i};
    site_imgs = dir(fullfile(img_folder, site, '/*.png'));
    for j = 1:length(site_imgs)
        name = replace(site_imgs(j).name, '.png', '');

        % image name
        img_path = fullfile(img_folder, site, sprintf('%s.png', name));

        % bounding box
        bb = csvread(fullfile(img_folder, site, sprintf('%s.csv', name)));
        bb = bb(any(bb ~= 0,2),:) % remove rows with only zeros

        % add to table
        training_img = [training_img; {img_path, bb}];
    end
end
head(training_img)
size(training_img, 1)




%% Train the ACF detector
acfDetector = trainACFObjectDetector(training_img,'NegativeSamplesFactor',10, 'NumStages', 4);




%% Plot an example image of the training set and the acf's prediction
nr = 50
img = imread(training_img{nr,1}{1});
annotation = 'bb';
for i=1:size(training_img{nr,2}{1}, 1)
    img = insertObjectAnnotation(img,'rectangle',training_img{nr,2}{1}(i,:),annotation);
end
imshow(img)

%Test the ACF detector on the same image.
img = imread(training_img{nr,1}{1});
[bboxes,scores] = detect(acfDetector,img, 'MinSize', [30,30], 'MaxSize', [70,70]);
%Display the detection results and insert the bounding boxes for objects into the image.
score_threshold = 40
for i = 1:length(scores)
   if scores(i)>score_threshold
       annotation = sprintf('Confidence = %.1f',scores(i));
       img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
   end
end
figure
imshow(img)




%% Evaluation
%% Create dataset with test images 
test_img = cell2table(cell(0,2), "VariableNames", ["path","AABB"]);
for i = 1:length(testsites)
    site = testsites{i};
    site_imgs = dir(fullfile(img_folder, site, '/*.png'));
    for j = 1:length(site_imgs)
        name = replace(site_imgs(j).name, '.png', '');

        % image name
        img_path = fullfile(img_folder, site, sprintf('%s.png', name));

        % bounding box
        if isfile(fullfile(img_folder, site, sprintf('%s.csv', name)))
            bb = csvread(fullfile(img_folder, site, sprintf('%s.csv', name)));
            bb = bb(any(bb ~= 0,2),:); % remove rows with only zeros
        else
            bb = {[]}
        end

        % add to table
        test_img = [test_img; {img_path, bb}];
    end
end
head(test_img)
size(test_img, 1)




%% Plot examples of test-images  and the acf's prediction
plot_pred(test_img, acfDetector)




%% Evaluate performance on all test images
score_threshold = 40
detections = cell2table(cell(0,2), "VariableNames", ["bboxes","scores"]);
gts = cell2table(cell(0,1), "VariableNames", ["person"]);
for i=1:size(test_img, 1)
    img = imread(test_img{i,1}{1});
    [bboxes,scores] = detect(acfDetector,img, 'MinSize', [30,30], 'MaxSize', [70,70]);
    bboxes = bboxes(scores > score_threshold, :);
    scores = scores(scores > score_threshold);

    detections = [detections; {bboxes, scores}];
    gts = [gts; {test_img{i,2}}];
end
head(detections)

% evaluate
addpath 'util'
iou_threshold = .10;
conf_threshold = .09;
averagePrecision = evaluateDetectionPrecision(detections,gts,iou_threshold)
[FP, TP, GT] = computeFpTpFn( detections, gts, iou_threshold, conf_threshold )

