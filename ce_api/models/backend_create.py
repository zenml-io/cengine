# coding: utf-8

"""
    maiot Core Engine API

    No description provided (generated by Swagger Codegen https://github.com/swagger-api/swagger-codegen)  # noqa: E501

    OpenAPI spec version: 0.1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class BackendCreate(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'args': 'object',
        'provider_id': 'str',
        'backend_class': 'str',
        'type': 'str',
        'name': 'str'
    }

    attribute_map = {
        'args': 'args',
        'provider_id': 'provider_id',
        'backend_class': 'backend_class',
        'type': 'type',
        'name': 'name'
    }

    def __init__(self, args=None, provider_id=None, backend_class=None, type=None, name=None):  # noqa: E501
        """BackendCreate - a model defined in Swagger"""  # noqa: E501
        self._args = None
        self._provider_id = None
        self._backend_class = None
        self._type = None
        self._name = None
        self.discriminator = None
        if args is not None:
            self.args = args
        if provider_id is not None:
            self.provider_id = provider_id
        if backend_class is not None:
            self.backend_class = backend_class
        if type is not None:
            self.type = type
        if name is not None:
            self.name = name

    @property
    def args(self):
        """Gets the args of this BackendCreate.  # noqa: E501


        :return: The args of this BackendCreate.  # noqa: E501
        :rtype: object
        """
        return self._args

    @args.setter
    def args(self, args):
        """Sets the args of this BackendCreate.


        :param args: The args of this BackendCreate.  # noqa: E501
        :type: object
        """

        self._args = args

    @property
    def provider_id(self):
        """Gets the provider_id of this BackendCreate.  # noqa: E501


        :return: The provider_id of this BackendCreate.  # noqa: E501
        :rtype: str
        """
        return self._provider_id

    @provider_id.setter
    def provider_id(self, provider_id):
        """Sets the provider_id of this BackendCreate.


        :param provider_id: The provider_id of this BackendCreate.  # noqa: E501
        :type: str
        """

        self._provider_id = provider_id

    @property
    def backend_class(self):
        """Gets the backend_class of this BackendCreate.  # noqa: E501


        :return: The backend_class of this BackendCreate.  # noqa: E501
        :rtype: str
        """
        return self._backend_class

    @backend_class.setter
    def backend_class(self, backend_class):
        """Sets the backend_class of this BackendCreate.


        :param backend_class: The backend_class of this BackendCreate.  # noqa: E501
        :type: str
        """

        self._backend_class = backend_class

    @property
    def type(self):
        """Gets the type of this BackendCreate.  # noqa: E501


        :return: The type of this BackendCreate.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this BackendCreate.


        :param type: The type of this BackendCreate.  # noqa: E501
        :type: str
        """

        self._type = type

    @property
    def name(self):
        """Gets the name of this BackendCreate.  # noqa: E501


        :return: The name of this BackendCreate.  # noqa: E501
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this BackendCreate.


        :param name: The name of this BackendCreate.  # noqa: E501
        :type: str
        """

        self._name = name

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(BackendCreate, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, BackendCreate):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other