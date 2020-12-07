function get_bb_pos(src,evt,num_bb,json_path)
    evname = evt.EventName;
    if evname=="ROIMoved"
        pos = evt.CurrentPosition;
        x_min = evt.CurrentPosition(1);
        y_min = evt.CurrentPosition(2);
        width = evt.CurrentPosition(3);
        height = evt.CurrentPosition(4);
        
        x_max = x_min+width;
        y_max = y_min+height;
        modified_label = [[x_max y_min];[x_min y_min];[x_min y_max];[x_max y_max]]
        
        json = readJSON(json_path);
        labels = json.Labels;
        labels(num_bb).poly = modified_label;
        json.Labels = labels;

        %jsonencode([json])
        json_str = jsonencode(json);
        fid = fopen(json_path, 'w+');
        fwrite(fid, json_str, 'char');
        fclose(fid);
    end
end
