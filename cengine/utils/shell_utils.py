def create_new_cell(contents):
    import sys
    from IPython.core.getipython import get_ipython

    if 'ipykernel' not in sys.modules:
        raise EnvironmentError('The magic functions are only usable in a '
                               'Jupyter notebook.')

    shell = get_ipython()

    payload = dict(
        source='set_next_input',
        text=contents,
        replace=False,
    )
    shell.payload_manager.write_payload(payload, single=False)
