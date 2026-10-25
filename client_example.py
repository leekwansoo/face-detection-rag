import requests
BASE="http://localhost:8000"
r=requests.get(f"{BASE}/capture/recognize", params={"k":3})
print(r.json())
