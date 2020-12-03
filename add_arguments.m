function args = add_arguments(arg_map, old_args)
new_args = {}
k = keys(arg_map)
v = values(arg_map)
for i = 1:length(arg_map)
    new_args = [new_args, k{i}, v{i}]
end
new_args
new_args{1}

args = [old_args; new_args]
end
