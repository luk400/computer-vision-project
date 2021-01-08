function out = augmentation_function(data, varargin)
    % Unpack original data.
    I = data{1};
    boxes = round(data{2});
    labels = data{3};
    
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
