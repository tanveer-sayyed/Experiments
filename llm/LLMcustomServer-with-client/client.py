from pprint import pprint
from requests import get

response = get(
             url="http://0.0.0.0:8000/ask",
             json={"prompt":"the quick brown fox jumped over the lazy dog sitting on the road."}
           )
pprint(response.json())
