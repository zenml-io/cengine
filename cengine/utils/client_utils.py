def filter_objects(o_list, **kwargs):
    if kwargs:
        o_list = [o for o in o_list
                  if all(getattr(o, k) == v for k, v in kwargs.items())]
    return o_list
