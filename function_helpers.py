def get_functions_by_name(function_names, functions):
    return [fn for fn in functions if fn.__name__ in function_names]

def get_function_names(functions):
    return list(map(lambda fn: fn.__name__, functions))