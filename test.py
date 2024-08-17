from lxml import etree
from io import StringIO
import requests

# Set explicit HTMLParser
parser = etree.HTMLParser()

page = requests.get('https://www.vlr.gg/371266/kr-esports-vs-cloud9-champions-tour-2024-americas-stage-2-ko/?game=178819&tab=overview')

# Decode the page content from bytes to string
html = page.content.decode("utf-8")

# Create your etree with a StringIO object which functions similarly
# to a fileHandler
tree = etree.parse(StringIO(html), parser=parser)

root = tree.getroot()

def print_node(node):
    # print(node.tag, " | ", node.attrib)
    if "class" in node.attrib and ("map" in node.attrib["class"] or "score " in node.attrib["class"]):
        print(node.text.strip())

    for child in node:
        print_node(child)
        
print_node(root)