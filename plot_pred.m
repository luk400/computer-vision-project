function plot_pred(test_img, acf_detector)

addpath 'util'
f = figure;

% panel for ui elements
panel = uipanel(f, 'Position',[0.01 0.02 0.98 0.4]);

%% create ui elements
% slider for selecting image
max_slider = size(test_img, 1);
c1 = uicontrol(panel, 'Style', 'slider',...
    'Min',1,'Max',max_slider,...
    'Units', 'Normalized',...
    'Position', [0.1 0.0 0.8 0.05],...
    'Callback', @plot_images,...
    'SliderStep', [1/(max_slider-1) 1/(max_slider-1)],...
    'Value', 3);
% text label
sub_panel1 = uipanel(panel,'Position',[0.1 0.05 0.8 0.15],'FontSize',11);
l1 = uicontrol(sub_panel1, 'style', 'text',... 
    'Units', 'Normalized',...
    'Position', [0 0 1 0.5]);

% slider for selecting score threshold
c2 = uicontrol(panel, 'Style', 'slider',...
    'Min',0,'Max',150,...
    'Units', 'Normalized',...
    'Position', [0.1 0.2 0.8 0.05],...
    'Callback', @plot_images,...
    'SliderStep', [1/149 1/149],...
    'Value', 40);
% text label
sub_panel2 = uipanel(panel,'Position',[0.1 0.25 0.8 0.15],'FontSize',11);
l2 = uicontrol(sub_panel2, 'style', 'text',... 
    'Units', 'Normalized',...
    'Position', [0 0 1 0.5]);

% slider for selecting minimum BB size
c3 = uicontrol(panel, 'Style', 'slider',...
    'Min',10,'Max',500,...
    'Units', 'Normalized',...
    'Position', [0.1 0.4 0.8 0.05],...
    'Callback', @plot_images,...
    'SliderStep', [1/489 1/489],...
    'Value', 25);
% text label
sub_panel3 = uipanel(panel,'Position',[0.1 0.45 0.8 0.15],'FontSize',11);
l3 = uicontrol(sub_panel3, 'style', 'text',... 
    'Units', 'Normalized',...
    'Position', [0 0 1 0.5]);

% slider for selecting maximum BB size
c4 = uicontrol(panel, 'Style', 'slider',...
    'Min',10,'Max',500,...
    'Units', 'Normalized',...
    'Position', [0.1 0.6 0.8 0.05],...
    'Callback', @plot_images,...
    'SliderStep', [1/489 1/489],...
    'Value', 50);
% text label
sub_panel4 = uipanel(panel,'Position',[0.1 0.65 0.8 0.15],'FontSize',11);
l4 = uicontrol(sub_panel4, 'style', 'text',... 
    'Units', 'Normalized',...
    'Position', [0 0 1 0.5]);

% slider for selecting maximum BB size
c5 = uicontrol(panel, 'Style', 'pushbutton',...
    'Units', 'Normalized',...
    'Position', [0.1 0.8 0.8 0.05],...
    'String', 'Calculate Precision on all test images',...
    'Callback', @calculate_precision);
% text label
sub_panel5 = uipanel(panel,'Position',[0.1 0.85 0.8 0.15],'FontSize',11);
l5 = uicontrol(sub_panel5, 'style', 'text',... 
    'Units', 'Normalized',...
    'Position', [0 0 1 0.5]);



% call function for plotting at initialization of the gui
plot_images()

    % define function for plotting images,
    % this function is called every time a ui-element is used 
    function plot_images(src,event)
        % get values from ui-elements
        nr = round(c1.Value);
        score_threshold = round(c2.Value);
        min_bb = round(c3.Value);
        max_bb = max(round(c4.Value),min_bb+1);
        set(l1, 'String', sprintf('Viewing image number %d out of %d',nr,size(test_img,1)));
        set(l2, 'String', sprintf('Score threshold: %d',score_threshold));
        set(l3, 'String', sprintf('Min. BB side size: %d px',min_bb));
        set(l4, 'String', sprintf('Max. BB side size: %d px',max_bb));

        
        % first plot: image with true bb
        img = imread(test_img{nr,1}{1}); % read image
        annotation = ' ';
        for i=1:size(test_img{nr,2}{1}, 1) % draw bounding boxes
            img = insertObjectAnnotation(img,'rectangle',test_img{nr,2}{1}(i,:),annotation,'LineWidth',5);
        end
        subplot(1,2,1); imshow(img); title("True BB's"); 
        
        % second plot: image with predicted bb
        img = imread(test_img{nr,1}{1}); % read image
        [bboxes,scores] = detect(acf_detector,img, 'MinSize', [min_bb,min_bb], 'MaxSize', [max_bb,max_bb]); % get labels and scores from acf detector
        %Display the detection results and insert the bounding boxes for objects into the image.
        for i = 1:length(scores) % draw bounding boxes
           if scores(i)>score_threshold % draw only if score is above threshold
               annotation = sprintf('Confidence = %.1f',scores(i));
               img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation,'LineWidth',5);
           end
        end
        subplot(1,2,2); imshow(img); title("Predictions");

    end

    function calculate_precision(src,event)
        set(l5, 'String', 'Calculating Precision...');

        score_threshold = round(c2.Value);
        min_bb = round(c3.Value);
        max_bb = max(round(c4.Value),min_bb+1);

        detections = cell2table(cell(0,2), "VariableNames", ["bboxes","scores"]);
        gts = cell2table(cell(0,1), "VariableNames", ["person"]);
        for i=1:size(test_img, 1)
            img = imread(test_img{i,1}{1});
            [bboxes,scores] = detect(acf_detector,img, 'MinSize', [min_bb,min_bb], 'MaxSize', [max_bb,max_bb]);
            bboxes = bboxes(scores > score_threshold, :);
            scores = scores(scores > score_threshold);
        
            detections = [detections; {bboxes, scores}];
            gts = [gts; {test_img{i,2}}];
        end
        
        % evaluate
        iou_threshold = .10;
        conf_threshold = .09;
        averagePrecision = evaluateDetectionPrecision(detections,gts,iou_threshold);
        [FP, TP, GT] = computeFpTpFn( detections, gts, iou_threshold, conf_threshold );

        set(l5, 'String', sprintf('Threshold=%d, Min.BB=%d, Max.BB=%d | Precision: %.2f%% | FP: %d | TP: %d | GT: %d',...
            score_threshold, min_bb, max_bb, averagePrecision*100,FP,TP,GT));

    end

end


