function [detections, gts] = create_detections_and_gts_tables(img_folder, predictions_path, annotations_path)

%# %%
testsites = { 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'};
test_img = cell2table(cell(0,2), "VariableNames", ["path","AABB"]);
for i = 1:length(testsites)
    site = testsites{i};
    site_imgs = dir(fullfile(img_folder, site, '/*.png'));
    for j = 1:length(site_imgs)
        name = replace(site_imgs(j).name, '.png', '');

        % image name
        img_name = sprintf('%s.png', name);

        % bounding box
        if isfile(fullfile(img_folder, site, sprintf('%s.csv', name)))
            bb = csvread(fullfile(img_folder, site, sprintf('%s.csv', name)));
            bb = bb(any(bb ~= 0,2),:); % remove rows with only zeros
        else
            bb = {[]}
        end

        % add to table
        test_img = [test_img; {img_name, bb}];
    end
end

pred_json = readJSON(predictions_path);
ann_json = readJSON(annotations_path);

%# %%

detections = cell2table(cell(0,2), "VariableNames", ["bboxes","scores"]);
gts = cell2table(cell(0,1), "VariableNames", ["person"]);

% no idea why, but I have to initialize first row with double.empty() entries
% otherwise I get an error in the following loop. I remove this row again 
% after the loop.
detections = [detections; {double.empty(0,4), double.empty(0,1)}];

for i=1:size(test_img, 1)
    img_name = test_img{i,1}{1};

    % the following is VERY inefficient, but doesn't matter for now, it works
    % loop over all entries in pred_json
    bboxes = double.empty(0,4);
    scores = double.empty(0,1);
    for j=1:size(pred_json,1) 
        % get image id of current entry 
        pred_img_id = pred_json(j).image_id; % get img_
        % get image file name corresponding to the image id (pred_img_id+1) because
        % we img_id starts at 0
        pred_img_name = ann_json.images(pred_img_id+1).file_name;

        % if image name of current entry matches the one of the current
        % test image, add bbox to detections table 
        if strcmp(img_name,pred_img_name)
            new_row = transpose(pred_json(j).bbox);
            score = pred_json(j).score;
            bboxes = [bboxes; new_row];
            scores = [scores; score];
        end
    end

    % add to detections and gts tables
    detections = [detections; {bboxes, scores}];
    gts = [gts; {test_img{i,2}}];

end
detections = cell2table(detections{2:end,:}, "VariableNames", ["bboxes","scores"]);
%# %%


end
