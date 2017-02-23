# coding: utf-8

import simplejson


def json_response(error, data):
    response = {}
    if error:
        response["error"] = error
    response["data"] = data
    return simplejson.dumps(response)


def error(code, message):
    return {
        "code":code,
        "message":message
    }


