from flask import Flask
from flask import request

from namegen import NameGenerator

app = Flask(__name__)

namegen = NameGenerator()


@app.route('/', methods=['POST'])
def get_name():
    return namegen.get_name_for(request.form['source'])