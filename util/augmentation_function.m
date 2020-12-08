function out = augmentation_function(data, varargin)
    % Unpack original data.
    I = data{1};
    boxes = round(data{2});
    labels = data{3};
    
    % Apply random color jitter. only for rgb
    %I = jitterColorHSV(I,"Brightness",0.3,"Contrast",0.4,"Saturation",0.2);
    
    % Define random affine transform.
    tform = randomAffine2d(varargin{:});
    rout = affineOutputView(size(I),tform);
    
    % Transform image and bounding box labels.
    augmentedImage = imwarp(I,tform,"OutputView",rout);
    [augmentedBoxes, valid] = bboxwarp(boxes,tform,rout,'OverlapThreshold',0.4);
    augmentedLabels = labels(valid);
    
    % Return augmented data.
    out = {augmentedImage,augmentedBoxes,augmentedLabels};
end
