import ce_api

# config = ce_api.Configuration()
# config.host = 'http://127.0.0.1:8000'
# api_instance = ce_api.LoginApi(ce_api.ApiClient(config))
# print(api_instance.login_access_token_api_v1_login_access_token_post(
#     username='hamza+dev@maiot.io', password='testtest'))

config = ce_api.Configuration()
config.host = '34.77.41.143:13703'

api_instance = ce_api.LoginApi(ce_api.ApiClient(config))
output = api_instance.login_access_token_api_v1_login_access_token_post(
    username='hamza+dev@maiot.io', password='testtest')

config.access_token = output.access_token

org_api = ce_api.OrganizationsApi(ce_api.ApiClient(config))
print(org_api.get_loggedin_organization_api_v1_organizations_get())
