const activation_functions = {
  "relu": (x) => Math.max(0, x),
  "sigmoid": (x) => 1 / (1 + Math.exp(-x)),
  "tanh": (x) => Math.tanh(x),
  "softmax": (x) => {
    const expVals = x.map(v => Math.exp(v - Math.max(...x)));
    const sum = expVals.reduce((a, b) => a + b, 0);
    return expVals.map(v => v / sum);
  },
  "leaky_relu": (x, alpha = 0.01) => x > 0 ? x : alpha * x,
  "elu": (x, alpha = 1.0) => x > 0 ? x : alpha * (Math.exp(x) - 1),
  "swish": (x) => x * this.sigmoid(x),
  "gaussian": (x) => Math.exp(-x ** 2),
  "softplus": (x) => Math.log(1 + Math.exp(x)),
  "null": (x) => x
};

class Model {
  constructor(layer_sizes, activations = null) {
    this.activations = activations ? activations.map(act => activation_functions[act]) : Array(layer_sizes.length - 1).fill(null);
    this.act_str = activations;
    this.weights = Array.from({ length: layer_sizes.length - 1 }, () =>
      Array.from({ length: layer_sizes[i] }, () =>
        Array.from({ length: layer_sizes[i + 1] }, () => Math.random())
      )
    );
  }

  forward(X) {
    this.a = [X];
    for (let i = 0; i < this.weights.length; i++) {
      X = X.map(x => this.dot(x, this.weights[i]));
      if (this.activations[i]) {
        X = X.map(this.activations[i]);
      }
      this.a.push(X);
    }
    return X;
  }

  predict(X) {
    return X.map(x => this.forward(x));
  }

  dot(a, b) {
    return a.reduce((acc, val, i) => acc + val * b[i], 0);
  }
}

function load_model(path) {
  const data = JSON.parse(fs.readFileSync(path, 'utf8'));
  const act_str = data.activations;
  const activations = act_str.map(act => activation_functions[act]);
  const weights = data.weights.map(w => w.map(row => row.map(Number)));
  const model = new Model();
  model.weights = weights;
  model.activations = activations;
  return model;
}

