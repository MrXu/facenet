# coding: utf-8

from flask import jsonify


def json_response(error, data):
    response = {}
    if error:
        response["error"] = error
    response["data"] = data
    return jsonify(response)


def error(code, message):
    return {
        "code":code,
        "message":message
    }


