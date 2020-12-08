function relabeling_func(imgs)
    addpath 'util'
    %% axis-aligned bounding box labels
    f = figure;
    panel = uipanel(f, 'Position',[0.01 0.02 0.98 0.2]);
    % slider for selecting image
    max_slider = size(imgs, 1);
    c1 = uicontrol(panel, 'Style', 'slider',...
        'Min',1,'Max',max_slider,...
        'Units', 'Normalized',...
        'Position', [0.1 0.0 0.8 0.05],...
        'Callback', @plot_images,...
        'SliderStep', [1/(max_slider-1) 1/(max_slider-1)],...
        'Value', 1);
    % text label
    sub_panel1 = uipanel(panel,'Position',[0.1 0.05 0.8 0.15],'FontSize',11);
    l1 = uicontrol(sub_panel1, 'style', 'text',... 
        'Units', 'Normalized',...
        'Position', [0 0 1 0.5]);
        % button for adding BB
    button=uicontrol(panel,'Style','pushbutton',...
        'String','Save modified BB','Units','normalized',...
        'Position',[0.1 0.4 0.2 0.2],'Visible','on',...
        'Callback', @save_BB);
    % button for saving modified BB
    button=uicontrol(panel,'Style','pushbutton',...
        'String','Add BB','Units','normalized',...
        'Position',[0.1 0.7 0.2 0.2],'Visible','on',...
        'Callback', @add_BB);

    
    % call function for plotting at initialization of the gui
    plot_images()
    
    % define function for plotting images,
    % this function is called every time a ui-element is used 
    function plot_images(src,event)
        img_nr = round(c1.Value);
        set(l1, 'String', sprintf('Viewing image number %d out of %d',...
            img_nr,size(imgs,1)));
   
        img = imread(imgs{img_nr,1}{1}); % read image

        global json_path
        json_path = imgs{img_nr,3}{1};
        global roi_objects
        roi_objects = [];

        imshow(img);
        title(imgs{img_nr,1}, 'Interpreter', 'none');
        json = readJSON(json_path);
        labels = json.Labels;
        
        % draw AABBs 
        img_size = [512 640]; % size of the images
        if ~isempty(labels) && ~isempty({labels.poly})
            [absBBs, relBBs, ~] = saveLabels( {labels.poly}, img_size, [] );
            
            for i_proj = 1:size(absBBs,1)
                x_min = absBBs(i_proj,1); 
                x_max = absBBs(i_proj,2); 
                y_min = absBBs(i_proj,3); 
                y_max = absBBs(i_proj,4);
                pos_rect = [x_min, y_min, x_max-x_min, y_max-y_min];
                % add rectangle-roi to list of roi's
                roi_objects = [roi_objects; ...
                    drawrectangle('Position',pos_rect, ...
                    'Color', 'yellow', 'Label', sprintf('bb%d', i_proj))];
            end
        end
    end

    function add_BB(src,event)
        global roi_objects
        bb_nr = size(roi_objects,1)+1;
        pos_rect = [231, 320, 50, 50];
        roi_objects = [roi_objects; ...
                    drawrectangle('Position',pos_rect, ...
                    'Color', 'red', 'Label', sprintf('bb%d', bb_nr))];
    end

    function save_BB(src,event)
        sprintf('save modified bounding boxes...')

        global roi_objects
        global json_path

        json = readJSON(json_path);
        labels = json.Labels;

        num_label = 1;
        for num_roi=1:size(roi_objects,1)
            roi = roi_objects(num_roi);
            if isvalid(roi) % if it wasn't deleted
                % get new position
                pos = roi.Position;
                x_min = pos(1);
                y_min = pos(2);
                width = pos(3);
                height = pos(4);
                
                x_max = x_min+width;
                y_max = y_min+height;

                % new label
                labels(num_label).poly = [[x_max y_min];[x_min y_min];[x_min y_max];[x_max y_max]];

                num_label = num_label + 1;
            else
                labels(num_label) = [];
            end
        end
        % overwrite original labels in json with modified labels
        json.Labels = labels;

        % save json with the modified bb
        json_str = jsonencode(json);
        fid = fopen(json_path, 'w+');
        fwrite(fid, json_str, 'char');
        fclose(fid);
    end
end

