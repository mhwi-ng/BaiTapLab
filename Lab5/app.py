from flask import Flask, render_template, Response
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend 'Agg' cho Matplotlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from cvxopt import matrix as cvxopt_matrix, solvers as cvxopt_solvers

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_plot1():
    # Code cho biểu đồ đầu tiên
    x = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694], [0.176, 0.458], 
                  [0.306, 0.753], [0.936, 0.413], [0.215, 0.410], [0.612, 0.375], [0.784, 0.602], 
                  [0.612, 0.554], [0.357, 0.254], [0.204, 0.775], [0.512, 0.745], [0.498, 0.287], 
                  [0.251, 0.557], [0.502, 0.523], [0.119, 0.687], [0.495, 0.924], [0.612, 0.851]])
    y = np.array([-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1])
    y = y.astype('float').reshape(-1, 1)

    m, n = x.shape
    X_dash = y * x
    H = np.dot(X_dash , X_dash.T) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))

    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    w = ((y * alphas).T @ x).reshape(-1,1)
    S = (alphas > 1e-4).flatten()

    b = y[S] - np.dot(x[S], w)
    b = np.mean(b)

    x1, x2 = -1, 2
    y1 = (-w[0] * x1 - b) / w[1]
    y2 = (-w[0] * x2 - b) / w[1]

    plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), s=50, cmap='autumn')
    plt.plot([x1, x2], [y1, y2], 'k-')
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def generate_plot2():
    # Code cho biểu đồ thứ hai

    N = 1000
    X = np.random.multivariate_normal([2,2], np.eye(2), N)
    X = np.vstack((X, np.random.multivariate_normal([-2,-2], np.eye(2), N)))
    y = np.hstack((np.ones(N), -np.ones(N)))

    y = y.astype('float').reshape(-1, 1)

    m, n = X.shape
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.

    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(-np.eye(m))
    h = cvxopt_matrix(np.zeros(m))

    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    w = ((y * alphas).T @ X).reshape(-1,1)
    S = (alphas > 1e-4).flatten()

    b = y[S] - np.dot(X[S], w)
    b = np.mean(b)

    x1, x2 = -5, 5
    y1 = (-w[0] * x1 - b) / w[1]
    y2 = (-w[0] * x2 - b) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=50, cmap='autumn')
    plt.plot([x1, x2], [y1, y2], 'k-')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

@app.route('/plot1.png')
def plot1():
    buf = generate_plot1()
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/plot2.png')
def plot2():
    buf = generate_plot2()
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
