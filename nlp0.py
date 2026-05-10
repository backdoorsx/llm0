import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import unicodedata
import re

class Examples_df:
    """
    Trieda obsahujúca príklady matematických funkcií a ich vizualizáciu.

    Účel:
        Demonštrácia funkcií, ktoré sa používajú pri učení modelov
        (napr. loss funkcie, non-convex povrchy).

    Obsah:
        - definícia funkcie f(x)
        - vykreslenie grafu
    """

    def ex_function(self, x):
        """
        Definícia funkcie f(x) = x^4 - 3x^3 + 2

        Vstup:
            x (float alebo np.array)

        Výstup:
            y = hodnota funkcie

        Matematika:
            f(x) = x^4 - 3x^3 + 2
        """
        return x**4 - 3*x**3 + 2

    def plot_function(self):
        """
        Vykreslí funkciu do grafu pomocou matplotlib.

        Vstup:
            žiadny

        Výstup:
            zobrazí graf

        Logika:
            1. vytvorí rozsah x
            2. spočíta y = f(x)
            3. vykreslí graf
        """

        x = np.linspace(-1, 3.5, 400)
        y = self.ex_function(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="f(x) = x^4 - 3x^3 + 2")

        plt.title("Non-convex funkcia")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)

        plt.grid(True)
        plt.legend()
        plt.show()


class Examples1D:
    """
    Trieda na vizualizáciu gradient descentu na 1D funkcii.
    """

    def f(self, x):
        """
        Funkcia f(x) = x^4 - 3x^3 + 2
        """
        return x**4 - 3*x**3 + 2

    def df(self, x):
        """
        Derivácia funkcie (gradient)

        f(x) = x^4 - 3x^3 + 2
        f'(x) = 4x^3 - 9x^2
        """
        return 4*x**3 - 9*x**2

    def gradient_descent_path(self, x0, lr=0.01, steps=50):
        """
        Vypočíta trajektóriu bodu pri gradient descent.

        Vstup:
            x0 (float): štartovacia pozícia
            lr (float): learning rate
            steps (int): počet krokov

        Výstup:
            list hodnôt x
        """
        x = x0
        path = [x]

        for _ in range(steps):
            x = x - lr * self.df(x)
            path.append(x)

        return path

    def animate(self):
        """
        Vytvorí animáciu pohybu bodu po funkcii.
        """

        # rozsah funkcie
        x_vals = np.linspace(-1, 3.5, 400)
        y_vals = self.f(x_vals)

        # gradient descent trajektória
        path = self.gradient_descent_path(x0=1.5, lr=0.01, steps=40)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, y_vals, label="f(x)")

        point, = ax.plot([], [], "ro", label="gradient descent")

        ax.set_title("Gradient Descent – pohyb bodu po funkcii")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)

        def update(i):
            x = path[i]
            y = self.f(x)
            point.set_data([x], [y])
            return point,

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(path),
            interval=200,
            blit=True
        )

        plt.show()

class LossLandscape2D:
    """
    Non-konvexna loss funkcia + gradient + vizualizacia.

    Uloha:
        Simulacia prostredia s viacerymi minimami a maximami.

    Matematika:
        f(x,y) = sin(x)*cos(y) + 0.1(x^2 + y^2)
    """

    def f(self, x, y):
        """
        Loss funkcia.

        Vstup:
            x, y (float alebo np.array)

        Vystup:
            float alebo np.array
        """
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

    def grad(self, x, y):
        """
        Gradient funkcie (rucne derivovany).

        Vstup:
            x, y (float)

        Vystup:
            tuple (df/dx, df/dy)

        Matematika:
            d/dx = cos(x)cos(y) + 0.2x
            d/dy = -sin(x)sin(y) + 0.2y
        """

        dx = np.cos(x) * np.cos(y) + 0.2 * x
        dy = -np.sin(x) * np.sin(y) + 0.2 * y

        return dx, dy

    def run(self, x0, y0, lr=0.1, steps=30):
        """
        Gradient descent v 2D.

        Vstup:
            x0, y0 (float): start
            lr (float): learning rate
            steps (int): pocet krokov

        Vystup:
            np.array shape (steps, 2)
        """

        path = [(x0, y0)]
        x, y = x0, y0

        for _ in range(steps):
            gx, gy = self.grad(x, y)

            x = x - lr * gx
            y = y - lr * gy

            path.append((x, y))

        return np.array(path)

    def plot(self, path):
        """
        Vykresli loss landscape + trajektoriu.

        Vstup:
            path (np.array): body gradient descentu

        Vystup:
            graf
        """

        x = np.linspace(-5, 5, 300)
        y = np.linspace(-5, 5, 300)
        X, Y = np.meshgrid(x, y)

        Z = self.f(X, Y)

        plt.figure(figsize=(7, 7))

        # mapa "kopcov a jam"
        plt.contour(X, Y, Z, levels=30)

        # trajektoria optimalizacie
        plt.plot(path[:, 0], path[:, 1], 'ro-')

        plt.title("Non-convex loss landscape + GD path")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()


class LossLandscape3D:
    """
    3D vizualizacia loss funkcie + gradient descent.

    Uloha:
        Ukazat ako vyzera "kopec" v 3D priestore.
    """

    def f(self, x, y):
        """
        Loss funkcia.

        Vstup:
            x, y (float alebo np.array)

        Vystup:
            float alebo np.array
        """
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

    def grad(self, x, y):
        """
        Gradient funkcie.

        Vstup:
            x, y (float)

        Vystup:
            (dx, dy)
        """
        dx = np.cos(x) * np.cos(y) + 0.2 * x
        dy = -np.sin(x) * np.sin(y) + 0.2 * y
        return dx, dy

    def run(self, x0, y0, lr=0.1, steps=30):
        """
        Gradient descent.

        Vstup:
            x0, y0 (float)
            lr (float)
            steps (int)

        Vystup:
            np.array (N,2)
        """

        path = [(x0, y0)]
        x, y = x0, y0

        for _ in range(steps):
            gx, gy = self.grad(x, y)

            x = x - lr * gx
            y = y - lr * gy

            path.append((x, y))

        return np.array(path)

    def plot_3d(self, path):
        """
        3D vizualizacia loss surface + GD trajektorie.

        Vstup:
            path (np.array)
        """

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # grid
        x = np.linspace(-5, 5, 150)
        y = np.linspace(-5, 5, 150)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        # surface (kopec)
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)

        # trajektoria gradient descentu
        z_path = self.f(path[:, 0], path[:, 1])
        ax.plot(path[:, 0], path[:, 1], z_path, 'r.-', linewidth=2)

        ax.set_title("3D Loss Landscape + Gradient Descent")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("loss")

        plt.show()

class SGDvsGD:
    """
    Porovnanie gradient descent vs stochastic gradient descent.
    """

    def f(self, x, y):
        """
        Loss funkcia.

        Vstup:
            x, y (float)

        Vystup:
            float
        """
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)

    def grad(self, x, y):
        """
        Presny gradient (GD).

        Vstup:
            x, y

        Vystup:
            dx, dy
        """
        dx = np.cos(x) * np.cos(y) + 0.2 * x
        dy = -np.sin(x) * np.sin(y) + 0.2 * y
        return dx, dy

    def noisy_grad(self, x, y, noise=0.3):
        """
        Stochastic gradient (SGD simulacia).

        Pridavame hluk aby sme simulovali mini-batch.

        Vstup:
            x, y
            noise (float)

        Vystup:
            dx, dy
        """
        dx, dy = self.grad(x, y)

        dx += np.random.randn() * noise
        dy += np.random.randn() * noise

        return dx, dy

    def run(self, x0, y0, lr=0.1, steps=30, stochastic=False):
        """
        Spustenie GD alebo SGD.

        Vstup:
            stochastic (bool): ci pouzit noise
        """

        path = [(x0, y0)]
        x, y = x0, y0

        for _ in range(steps):

            if stochastic:
                gx, gy = self.noisy_grad(x, y)
            else:
                gx, gy = self.grad(x, y)

            x = x - lr * gx
            y = y - lr * gy

            path.append((x, y))

        return np.array(path)

    def plot(self):
        """
        Porovnanie GD vs SGD.
        """

        gd_path = self.run(3, 3, stochastic=False)
        sgd_path = self.run(3, 3, stochastic=True)

        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.f(X, Y)

        plt.figure(figsize=(7, 7))

        # mapa kopcov
        plt.contour(X, Y, Z, levels=30)

        # GD = hladka ciara
        plt.plot(gd_path[:, 0], gd_path[:, 1], 'b.-', label="GD (smooth)")

        # SGD = skakanie
        plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'ro-', label="SGD (noisy)")

        plt.legend()
        plt.title("GD vs SGD (prečo SGD 'skáče')")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()
        







import json
from collections import Counter
import numpy as np

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

def tokenize(text):
    return text.lower().split()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
def remove_diacritics(text):
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ASCII", "ignore").decode("ASCII")

def clean_text(text):
    text = text.lower()
    text = remove_diacritics(text)
    text = re.sub(r"[^\w\s]", " ", text)  # všetko okrem písmen/čísel → medzera
    text = re.sub(r"\s+", " ", text).strip()  # zjednotí medzery
    return text

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []

    for item in raw:
        text = item["text"]
        text = clean_text(text)
        data.append((text, None))

    return data

# --------------------------------------------------
# VOCAB
# --------------------------------------------------

def build_vocab(data, min_freq=1):

    counter = Counter()

    for text, _ in data:
        counter.update(tokenize(text))

    vocab = ["<PAD>", "<UNK>"]

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab.append(word)

    word2idx = {w:i for i, w in enumerate(vocab)}
    idx2word = {i:w for w, i in word2idx.items()}

    return word2idx, idx2word

# --------------------------------------------------
# ENCODE
# --------------------------------------------------

def encode_oov(tokens, word2idx):

    unk = word2idx["<UNK>"]

    return [
        word2idx.get(t, unk)
        for t in tokens
    ]

# --------------------------------------------------
# DATASET
# --------------------------------------------------

def build_language_dataset(data, word2idx):

    pairs = []

    for text, _ in data:

        tokens = tokenize(text)

        encoded = encode_oov(tokens, word2idx)

        if len(encoded) < 2:
            continue

        x = encoded[:-1]
        y = encoded[1:]

        pairs.append((x, y))

    return pairs

# --------------------------------------------------
# SOFTMAX
# --------------------------------------------------

def softmax(x):

    x = x - np.max(x)

    e = np.exp(x)

    return e / np.sum(e)

# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------

class Embedding:

    def __init__(self, vocab_size, dim, max_len=64):

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len

        # token embeddings
        self.E = np.random.randn(vocab_size, dim) * 0.01

        # positional embeddings
        self.P = np.random.randn(max_len, dim) * 0.01

    def forward(self, x):

        token_emb = self.E[x]

        positions = np.arange(len(x))

        pos_emb = self.P[positions]

        return token_emb + pos_emb
# --------------------------------------------------
# TRANSFORMER
# --------------------------------------------------

class MiniTransformerTrainable:

    def __init__(self, dim):

        self.dim = dim

        self.Wq = np.random.randn(dim, dim) * 0.01
        self.Wk = np.random.randn(dim, dim) * 0.01
        self.Wv = np.random.randn(dim, dim) * 0.01

    def softmax(self, x):

        x = x - np.max(x, axis=-1, keepdims=True)

        e = np.exp(x)

        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, x):

        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        scores = (Q @ K.T) / np.sqrt(self.dim)

        # causal mask
        n = scores.shape[0]

        mask = np.triu(np.ones((n, n)), k=1)

        scores = scores - 1e9 * mask

        attn = self.softmax(scores)

        out = attn @ V

        cache = (x, Q, K, V, attn)

        return out, cache

    def backward(self, dout, cache, lr=0.01):

        x, Q, K, V, attn = cache

        n = x.shape[0]

        dV = attn.T @ dout

        datt = dout @ V.T

        dscores = np.zeros_like(datt)

        for i in range(n):

            a = attn[i][:, None]
            da = datt[i][:, None]

            dscores[i] = (np.diagflat(a) - a @ a.T) @ da[:, 0]

        dQ = dscores @ K / np.sqrt(self.dim)

        dK = dscores.T @ Q / np.sqrt(self.dim)

        dWq = x.T @ dQ
        dWk = x.T @ dK
        dWv = x.T @ dV

        self.Wq -= lr * dWq
        self.Wk -= lr * dWk
        self.Wv -= lr * dWv

        dx = (
            dQ @ self.Wq.T +
            dK @ self.Wk.T +
            dV @ self.Wv.T
        )

        return dx

# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

class SimpleClassifierLLM:

    def __init__(self, dim, vocab_size):

        self.W = np.random.randn(dim, vocab_size) * 0.01

        self.b = np.zeros(vocab_size)

    def forward(self, h):

        logits = h @ self.W + self.b

        return softmax(logits)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

def train_lm_transformer(embedding, transformer, classifier, pairs, epochs=300, lr=0.01):

    for epoch in range(epochs):

        total_loss = 0

        for x, y in pairs:

            emb = embedding.forward(x)

            out, cache = transformer.forward(emb)

            seq_loss = 0

            dh_total = np.zeros_like(out)

            # GPT style loss
            for t in range(len(y)):

                h = out[t]

                logits = h @ classifier.W + classifier.b

                probs = softmax(logits)

                loss = -np.log(probs[y[t]] + 1e-9)

                seq_loss += loss

                # backward classifier
                dlogits = probs.copy()

                dlogits[y[t]] -= 1

                dW = np.outer(h, dlogits)

                db = dlogits

                dh = classifier.W @ dlogits

                classifier.W -= lr * dW
                classifier.b -= lr * db

                dh_total[t] = dh

            # backward transformer
            d_emb = transformer.backward(dh_total, cache, lr)

            # backward embedding
            for i, idx in enumerate(x):

                embedding.E[idx] -= lr * d_emb[i]

            total_loss += seq_loss / len(y)

        avg_loss = total_loss / len(pairs)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, loss={avg_loss:.4f}")

# --------------------------------------------------
# GENERATE
# --------------------------------------------------

def generate_transformer(embedding, transformer, classifier, text, word2idx, idx2word, max_len=20, temperature=1.0, top_k=5):

    words = tokenize(text)

    for _ in range(max_len):

        x = encode_oov(words, word2idx)

        emb = embedding.forward(x)

        out, _ = transformer.forward(emb)

        h = out[-1]

        logits = h @ classifier.W + classifier.b

        probs = softmax(logits / temperature)

        top_idx = np.argsort(probs)[-top_k:]

        top_probs = probs[top_idx]

        top_probs /= np.sum(top_probs)

        idx = np.random.choice(top_idx, p=top_probs)

        words.append(idx2word[idx])

    return " ".join(words)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    dimension = 64
    data = load_dataset("datasetllm2.json")
    print(len(data))
    input('>')

    word2idx, idx2word = build_vocab(data)
    print(len(word2idx))
    input('>>')

    pairs = build_language_dataset(data, word2idx)

    embedding = Embedding(len(word2idx), dimension, max_len=64)

    transformer = MiniTransformerTrainable(dimension)

    classifier = SimpleClassifierLLM(dimension, len(word2idx))

    train_lm_transformer(embedding, transformer, classifier, pairs, epochs=600, lr=0.001)

    print(generate_transformer(embedding, transformer, classifier, "ahoj ako", word2idx, idx2word, max_len=10))

    # 1D EXAMPLE
    #ex = Examples()
    ##ex.plot_function()
    #ex.animate()

    # 2D EXAMPLE
    #model = LossLandscape2D()
    #path = model.run(x0=3, y0=3, lr=0.1, steps=25)
    #model.plot(path)

    # 3D EXAMPLE
    #model = LossLandscape3D()
    #path = model.run(x0=3, y0=3, lr=0.1, steps=30)
    #model.plot_3d(path)

    # SGD vs GD EXAMPLE
    #model = SGDvsGD()
    #model.plot()
