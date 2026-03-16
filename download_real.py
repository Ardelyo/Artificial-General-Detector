import urllib.request
import os

url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open('tests/real_test_photo.png', 'wb') as out_file:
    out_file.write(response.read())

print("Downloaded real test photo successfully.")
