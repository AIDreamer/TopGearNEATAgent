import http, subprocess

c = http.client.HTTPConnection('localhost', 37979)
c.request('POST', '/return', '{}')
doc = c.getresponse().read()
print(doc)