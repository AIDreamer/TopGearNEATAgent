import subprocess
from bottle import run, post, request, response, get, route

@route('/screen', method='POST')
def feedback():
    print(request.body.read())

run(host='localhost', port=49999, debug=True)