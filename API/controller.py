# coding=utf-8
from traceback import format_exc
from flask_cors import CORS
from flask import Flask
from flask import request, jsonify

from annotator import Annotator

def valid_param(val):
    return val != None and len(str(val).strip()) > 0

print("Creating app")
app = Flask(__name__)
cors = CORS(app)
annotatr = None

def build_app(config_file):
    global annotatr
    annotatr = Annotator.from_config(config_file)
    return app

@app.route('/AnnotateEssays', methods=['GET'])
def annotate_essay_text():
    try:
        text = request.args.get("text")
        if not valid_param(text):
            return jsonify({"error": "No essay text entered!"})

        return jsonify(annotatr.annotate(text))
    except Exception as e:
        return jsonify({"error": format_exc()})


if __name__ == "__main__":
    try:
        import sys
        if len(sys.argv) < 2:
            raise Exception("Missing config file command line argument")

        config_file = sys.argv[1]
        app = build_app(config_file)
        app.run(debug=False, host='0.0.0.0', port=5001)  # host= '0.0.0.0' will expose externally on host's ip

    except Exception as e:
        print(format_exc())
