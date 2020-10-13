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

class UserUpdate(object):
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
        'email': 'str',
        'full_name': 'str',
        'organization_id': 'str',
        'n_pipelines_executed': 'int',
        'firebase_id': 'str'
    }

    attribute_map = {
        'email': 'email',
        'full_name': 'full_name',
        'organization_id': 'organization_id',
        'n_pipelines_executed': 'n_pipelines_executed',
        'firebase_id': 'firebase_id'
    }

    def __init__(self, email=None, full_name=None, organization_id=None, n_pipelines_executed=0, firebase_id=None):  # noqa: E501
        """UserUpdate - a model defined in Swagger"""  # noqa: E501
        self._email = None
        self._full_name = None
        self._organization_id = None
        self._n_pipelines_executed = None
        self._firebase_id = None
        self.discriminator = None
        if email is not None:
            self.email = email
        if full_name is not None:
            self.full_name = full_name
        if organization_id is not None:
            self.organization_id = organization_id
        if n_pipelines_executed is not None:
            self.n_pipelines_executed = n_pipelines_executed
        if firebase_id is not None:
            self.firebase_id = firebase_id

    @property
    def email(self):
        """Gets the email of this UserUpdate.  # noqa: E501


        :return: The email of this UserUpdate.  # noqa: E501
        :rtype: str
        """
        return self._email

    @email.setter
    def email(self, email):
        """Sets the email of this UserUpdate.


        :param email: The email of this UserUpdate.  # noqa: E501
        :type: str
        """

        self._email = email

    @property
    def full_name(self):
        """Gets the full_name of this UserUpdate.  # noqa: E501


        :return: The full_name of this UserUpdate.  # noqa: E501
        :rtype: str
        """
        return self._full_name

    @full_name.setter
    def full_name(self, full_name):
        """Sets the full_name of this UserUpdate.


        :param full_name: The full_name of this UserUpdate.  # noqa: E501
        :type: str
        """

        self._full_name = full_name

    @property
    def organization_id(self):
        """Gets the organization_id of this UserUpdate.  # noqa: E501


        :return: The organization_id of this UserUpdate.  # noqa: E501
        :rtype: str
        """
        return self._organization_id

    @organization_id.setter
    def organization_id(self, organization_id):
        """Sets the organization_id of this UserUpdate.


        :param organization_id: The organization_id of this UserUpdate.  # noqa: E501
        :type: str
        """

        self._organization_id = organization_id

    @property
    def n_pipelines_executed(self):
        """Gets the n_pipelines_executed of this UserUpdate.  # noqa: E501


        :return: The n_pipelines_executed of this UserUpdate.  # noqa: E501
        :rtype: int
        """
        return self._n_pipelines_executed

    @n_pipelines_executed.setter
    def n_pipelines_executed(self, n_pipelines_executed):
        """Sets the n_pipelines_executed of this UserUpdate.


        :param n_pipelines_executed: The n_pipelines_executed of this UserUpdate.  # noqa: E501
        :type: int
        """

        self._n_pipelines_executed = n_pipelines_executed

    @property
    def firebase_id(self):
        """Gets the firebase_id of this UserUpdate.  # noqa: E501


        :return: The firebase_id of this UserUpdate.  # noqa: E501
        :rtype: str
        """
        return self._firebase_id

    @firebase_id.setter
    def firebase_id(self, firebase_id):
        """Sets the firebase_id of this UserUpdate.


        :param firebase_id: The firebase_id of this UserUpdate.  # noqa: E501
        :type: str
        """

        self._firebase_id = firebase_id

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
        if issubclass(UserUpdate, dict):
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
        if not isinstance(other, UserUpdate):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other