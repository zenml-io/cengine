import click
from tabulate import tabulate

import ce_api
from ce_api.models import AuthEmail
from ce_cli.cli import cli
from ce_cli.utils import api_client, api_call
from ce_cli.utils import check_login_status, pass_info, Info
from ce_cli.utils import declare, confirmation
from ce_standards import constants


@cli.group()
@pass_info
def auth(info):
    """Authentication utilities of the Core Engine"""
    pass


@auth.command()
@pass_info
def login(info):
    """Login with your username and password"""
    username = click.prompt('Please enter your email', type=str)
    password = click.prompt('Please enter your password', type=str,
                            hide_input=True)

    # API instance
    config = ce_api.Configuration()
    config.host = constants.API_HOST
    api_instance = ce_api.LoginApi(ce_api.ApiClient(config))

    output = api_call(
        func=api_instance.login_access_token_api_v1_login_access_token_post,
        username=username,
        password=password
    )

    info[constants.ACTIVE_USER] = username
    declare('Login successful!')
    if username in info:
        info[username][constants.TOKEN] = output.access_token
    else:
        info[username] = {constants.TOKEN: output.access_token}

    info.save()


@auth.command()
@pass_info
def logout(info):
    """Log out of your account"""
    if click.confirm('Are you sure that you want to log out?'):
        click.echo('Logged out!')
        info[constants.ACTIVE_USER] = None
        info.save()


@auth.command()
@click.option('--all', 'r_all', is_flag=True, help='Flag to reset all users')
@pass_info
def reset(info, r_all):
    """Reset cookies"""
    if r_all:
        if click.confirm('Are you sure that you want to reset for all?'):
            info = Info()
            info.save()
            click.echo('Info reset!')
        else:
            click.echo('Reset aborted!')
    else:
        active_user = info[constants.ACTIVE_USER]

        if click.confirm('Are you sure that you want to reset info for '
                         '{}?'.format(active_user)):
            info[active_user] = {}
            info.save()
            click.echo('Info reset!')
        else:
            click.echo('Reset aborted!')

        info[active_user] = {}
        info.save()


@auth.command()
@pass_info
def reset_password(info):
    """Send reset password link to registered email address"""
    confirmation('Are you sure you want to reset your password? This will '
                 'trigger an email for resetting your password and '
                 'clear cookies.', abort=True)
    check_login_status(info)
    api = ce_api.UsersApi(api_client(info))
    user = api_call(api.get_loggedin_user_api_v1_users_me_get)
    api = ce_api.LoginApi(api_client(info))
    api_call(api.send_reset_pass_email_api_v1_login_email_resetpassword_post,
             AuthEmail(email=user.email))
    info[constants.ACTIVE_USER] = None
    info.save()
    declare("Reset password email sent to {}".format(user.email))


@auth.command()
@pass_info
def whoami(info):
    """Info about the account which is currently logged in"""
    check_login_status(info)
    api = ce_api.UsersApi(api_client(info))
    billing_api = ce_api.BillingApi(api_client(info))

    user = api_call(api.get_loggedin_user_api_v1_users_me_get)
    bill = api_call(billing_api.get_user_billing_api_v1_billing_users_user_id_get,
                    user_id=user.id)
    table = [{
        'Email': info[constants.ACTIVE_USER],
        'Full Name': user.full_name if user.full_name else '',
        'Pipelines Run': user.n_pipelines_executed,
        'Processed Datapoints total': bill.total_processed_datapoints,
        'Cost Total': bill.cost_total,
        'Processed Datapoints this Month':
            bill.processed_datapoints_this_month,
        'Cost This Month': bill.cost_this_month,
    }]
    click.echo(tabulate(table, headers='keys', tablefmt='presto'))


@auth.command()
@pass_info
def organization(info):
    """Info about the account which is currently logged in"""
    check_login_status(info)
    api = ce_api.OrganizationsApi(api_client(info))
    billing_api = ce_api.BillingApi(api_client(info))

    org = api_call(api.get_loggedin_organization_api_v1_organizations_get)
    bill = api_call(
        billing_api.get_organization_billing_api_v1_billing_organization_get)
    table = [{
        'Organization Name': org.name,
        'Processed Datapoints total': bill.total_processed_datapoints,
        'Cost Total': bill.cost_total,
        'Processed Datapoints this Month':
            bill.processed_datapoints_this_month,
        'Cost This Month': bill.cost_this_month,
    }]
    click.echo(tabulate(table, headers='keys', tablefmt='presto'))
