# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : exception.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""


class VAException(Exception):
    """
    Parent Class of SLA-VAE Exception
    """

    def __init__(self):
        super().__init__()


class CommonException(VAException):
    """
    Common Util Exception
    """

    def __init__(self, err):
        super(CommonException, self).__init__()
        self.err = err
        self.error_code = 1

    def __str__(self):
        return f'COMMON UTILS ERROR: {self.err}.'


class FeatureException(VAException):
    """
    Feature Extraction Exception
    """

    def __init__(self, err):
        super(FeatureException, self).__init__()
        self.err = err
        self.error_code = 2

    def __str__(self):
        return f'FEATURE ERROR: {self.err}.'


class ModelException(VAException):
    """
    Model Exception
    """

    def __init__(self, err):
        super(ModelException, self).__init__()
        self.err = err
        self.error_code = 3

    def __str__(self):
        return f'MODEL ERROR: {self.err}.'
