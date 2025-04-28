import re
import json
import textwrap

# Helper function to wrap text
def wrap_text(text, width=80):
    return "<br>".join(textwrap.wrap(text, width=width))



def parse_json_safe(response_text):
    try:
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_substring = response_text[json_start:json_end]
        return json.loads(json_substring)
    except Exception as e:
        print(f"JSON parsing failed: {e}\nRaw response: {response_text}")
        return None