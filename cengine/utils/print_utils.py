import enum
import pprint

import yaml
from dateutil import tz


class PrintStyles(enum.Enum):
    NATIVE = 0
    YAML = 1
    PPRINT = 2


def to_pretty_string(d, style=PrintStyles.YAML):
    if style == PrintStyles.YAML:
        return yaml.dump(d, default_flow_style=False)
    elif style == PrintStyles.PPRINT:
        return pprint.pformat(d)
    else:
        return str(d)


def format_date(dt, format='%Y-%m-%d %H:%M:%S'):
    if dt is None:
        return ''
    local_zone = tz.tzlocal()
    # make sure this is UTC
    dt = dt.replace(tzinfo=tz.tzutc())
    local_time = dt.astimezone(local_zone)
    return local_time.strftime(format)


def format_timedelta(td):
    if td is None:
        return ''
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
