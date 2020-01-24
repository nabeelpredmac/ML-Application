## Flask authentication for bokeh

from functools import wraps
from flask import request, Response, redirect, Flask,render_template
from bokeh.util import session_id

app = Flask(__name__)

def check_auth(username, password):
    return username == 'xxxxx' and password == 'xxxxx'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

# Route for handling the login page logic
@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'xxxxxx':
            error = 'Invalid Credentials. Please try again.'
        else:
            s_id = session_id.generate_session_id()
            return redirect("http://192.168.0.99:5006/CRM_bokeh_app?bokeh-session-id={}".format(s_id), code=302)
    return render_template('login.html', error=error)


# def requires_auth(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         auth = request.authorization
#         if not auth or not check_auth(auth.username, auth.password):
#             return authenticate()
#         return f(*args, **kwargs)
#     return decorated

# @app.route('/')
# @requires_auth
# def redirect_to_bokeh():
#     s_id = session_id.generate_session_id()
#     return redirect("http://192.168.0.99:5006/CRM_bokeh_app?bokeh-session-id={}".format(s_id), code=302)

if __name__ == "__main__":
    app.run(host='192.168.0.99',port=5000)    