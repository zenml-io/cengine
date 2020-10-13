from ce_api.rest import ApiException


def api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ApiException as e:
        raise ApiException('{}: {}'.format(e.reason, e.body))
    except Exception as e:
        raise Exception('There is something wrong going on. Please contact '
                        'core@maiot.io to get further information.')


def find_closest_uuid(substr: str, options):
    candidates = [x.id for x in options if x.id.startswith(substr)]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise ValueError('No matching IDs found!')
    raise ValueError('Too many matching IDs.')


def resolve_workers(distributed, workers, cpus_per_worker):
    if distributed:
        if workers is None or cpus_per_worker is None:
            workers = 5
            cpus_per_worker = 4
    else:
        assert workers is None and cpus_per_worker is None, \
            'If you want to run your pipeline in a distributed setting,' \
            'please use "--distributed" or "-d"'
        workers = 1
        cpus_per_worker = 1

    return workers, cpus_per_worker


def format_uuid(uuid: str):
    return uuid[0:8]
