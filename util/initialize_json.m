function [training_json, test_json] = initialize_json()
    info_struct = struct('description', '', 'url', '', 'version', '', 'year', '', 'contributor', '', 'date_created', '');
    licenses_struct = struct('id', 1, 'name', nan, 'url', nan);
    categories_struct = struct('id', 1, 'name', 'person', 'supercategory', 'None');
    images_structs = [];
    annotations_structs = [];
    
    % create json struct
    json_struct = struct('info', info_struct, 'licenses', [], 'categories',...
        [], 'images', images_structs, 'annotations', annotations_structs);
    % need to specify the following 2 extra, otherwise jsonencode loses the
    % square brackets, which will cause an error in efficientdet parsing
    % (see https://stackoverflow.com/questions/46198670/using-jsonencode-with-length-1-array)
    json_struct.categories = {categories_struct}; 
    json_struct.licenses = {licenses_struct};

    training_json = json_struct;
    test_json = json_struct;
end
