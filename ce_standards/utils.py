# TODO: this needs to be imported from gdp directly and cli needs to
#   check for the installation for gdp


class ConfigKeys:
    @classmethod
    def get_keys(cls):
        keys = {key: value for key, value in cls.__dict__.items() if
                not isinstance(value, classmethod) and
                not isinstance(value, staticmethod) and
                not callable(value) and
                not key.startswith('__')}

        required = {k: v for k, v in keys.items() if not k.endswith('_')}
        optional = {k: v for k, v in keys.items() if k.endswith('_')}

        return required, optional

    @classmethod
    def key_check(cls, _input):
        # Check whether the given config is a dict
        assert isinstance(_input, dict), 'Please specify a dict for {}' \
            .format(cls.__name__)

        # Required and optional keys for the config dict
        required, optional = cls.get_keys()

        # Check for missing keys
        missing_keys = [k for k in required.values() if
                        k not in _input.keys()]
        assert len(missing_keys) == 0, \
            'Missing key(s) {} in {}'.format(missing_keys,
                                             cls.__name__)

        # Check for unknown keys
        unknown_keys = [k for k in _input.keys() if
                        k not in required.values() and
                        k not in optional.values()]
        assert len(unknown_keys) == 0, \
            'Unknown key(s) {} in {}. Required keys : {} ' \
            'Optional Keys: {}'.format(unknown_keys,
                                       cls.__name__,
                                       list(required.values()),
                                       list(optional.values()))
