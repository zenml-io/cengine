def serialize(obj):
    """
    Generalized function to turn cengine models into a serializable and
    representable format

    :param obj: The object to serialize
    :return result: The converted object
    """
    # For a list, convert each element and return another list
    if isinstance(obj, list):
        result = []
        for x in obj:
            if hasattr(x, 'to_serial'):
                v = x.to_serial()
            else:
                v = serialize(x)
            if v is not None:
                result.append(v)

        return result

    # For a dict, convert each value and save another dict with the same keys
    elif isinstance(obj, dict):
        result = {}
        for x, y in obj.items():
            if hasattr(y, 'to_serial'):
                v = y.to_serial()
            else:
                v = serialize(y)

            if v is not None:
                result.update({x: v})
        return result

    # For cengine models, get the attributes and convert its values
    elif hasattr(obj, 'get_attributes'):
        result = {}
        for attr in obj.get_attributes():
            value = getattr(obj, attr)

            if hasattr(value, 'to_serial'):
                v = value.to_serial()
            else:
                v = serialize(value)

            if v is not None:
                result.update({attr: v})
        return result

    # The remaining data structures
    else:
        return obj
