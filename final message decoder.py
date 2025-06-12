#Got this decoder from https://gist.github.com/jacquerie/cfb8a56636e2b9e12f51?permalink_comment_id=2687597#gistcomment-2687597
import base64

MESSAGE = '''
LF4SARcwJhIBV0VUVEIVJRwAAFN/Y0YRHwkCEQQVIhxGVE5zZAQBBAALGQAWcFVBUxE1JQ4ABBZJ VF9ScBAPFwY2JwgQHABJWEVVNhoJHRElJgwXHhFJVF9ScAwPGBswKAQWV0lOUxcTNRsIAAd0Y1tS VxYPEgBVe1lGEhs8ZEFIUEIZHQtTcAQ=
'''

KEY = 'WyattSCarpenter'

result = []
for i, c in enumerate(base64.b64decode(MESSAGE)):
  result.append(chr(c ^ ord(KEY[i % len(KEY)])))

print(''.join(result))