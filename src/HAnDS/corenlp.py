import json
import requests
import urllib.parse
class CoreNlPClient:

    def __init__(self, serverurl="http://127.0.0.1:9000/", annotators="tokenize, ssplit, pos"):

        self.properties = {}
        self.properties["annotators"] = annotators
        self.properties["tokenize.whitespace"] = False
        self.properties["tokenize.whitespace"] = False
        self.properties["tokenize.options"] = "normalizeParentheses=false,normalizeOtherBrackets=false,asciiQuotes=true,splitHyphenated=true"
        self.properties["outputFormat"] = "json"
        self.serverurl = serverurl


    def annotate(self, s):
        properties = json.dumps(self.properties)
        r = requests.post("%s?properties=%s" %(self.serverurl, properties), data=s)

        if r.status_code == 200:
            try:
                return json.loads(urllib.parse.unquote(r.text), strict=False)
            except json.decoder.JSONDecodeError:
                return None
        else:
            raise RuntimeError("%s \t %s"%(r.status_code, r.reason))
