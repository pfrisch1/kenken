import xml.etree.ElementTree as ET
import requests

def get_n_by_n_games(n):
	xml_request = requests.get("http://www.webkendoku.com/ref/gamedata%s.xml" % n)
	xml = xml_request.text
	with open("data/gamedata%s.xml" % n, "w") as f:
		f.write(xml)

if __name__ == "__main__":
	for i in [4, 6, 8]:
		get_n_by_n_games(i)