// ULY MURA SCAN — распознавание без сторонних ИИ-сервисов.
// Открытая библиотека TensorFlow.js + предобученная MobileNet v1 (0.25, Apache 2.0) как «глаз»,
// поверх неё — собственный классификатор, обученный на наших фото на странице train.html.
window.ML = (() => {
  const SIZE = 224;
  let base = null;

  async function loadBase(url = "model/mobilenet/model.json") {
    if (base) return base;
    const m = await tf.loadLayersModel(url);
    base = tf.model({ inputs: m.inputs, outputs: m.getLayer("global_average_pooling2d_1").output });
    tf.tidy(() => base.predict(tf.zeros([1, SIZE, SIZE, 3]))); // прогрев
    return base;
  }

  // Квадратный кадр из центра (как в рамке сканера) → 224×224, значения [-1, 1].
  // opts: zoom (<1 — ближе), dx/dy (сдвиг в долях), flip (зеркало) — для аугментации при обучении.
  function toInput(el, opts = {}) {
    return tf.tidy(() => {
      const img = tf.browser.fromPixels(el).toFloat().expandDims(0);
      const [, h, w] = img.shape;
      const s = Math.min(h, w) * (opts.zoom || 1);
      const cy = h / 2 + (opts.dy || 0) * s, cx = w / 2 + (opts.dx || 0) * s;
      const clamp = v => Math.max(0, Math.min(1, v));
      const box = [[clamp((cy - s / 2) / h), clamp((cx - s / 2) / w), clamp((cy + s / 2) / h), clamp((cx + s / 2) / w)]];
      let t = tf.image.cropAndResize(img, box, [0], [SIZE, SIZE]);
      if (opts.flip) t = tf.reverse(t, 2);
      return t.div(127.5).sub(1);
    });
  }

  // Вектор признаков 256 чисел, нормированный по длине.
  function embed(el, opts) {
    return tf.tidy(() => {
      const f = base.predict(toInput(el, opts)).reshape([1, -1]);
      return f.div(f.norm("euclidean", 1, true).add(1e-6));
    });
  }

  // Классификатор из classifier.json: { labels, W: [D][C], b: [C] }
  function makeClassifier(json) {
    const W = tf.tensor2d(json.W), b = tf.tensor1d(json.b);
    return {
      labels: json.labels, meta: json,
      async predict(el) {
        const p = tf.tidy(() => embed(el).matMul(W).add(b).softmax());
        const probs = await p.data(); p.dispose();
        return json.labels.map((className, i) => ({ className, probability: probs[i] }))
          .sort((a, c) => c.probability - a.probability);
      },
    };
  }

  // Обучение: X — массив векторов, y — индексы классов. Балансировка классов повтором примеров.
  async function train(X, y, nClasses, { epochs = 1000, lr = 0.02, l2 = 1e-5, onEpoch } = {}) {
    const counts = Array(nClasses).fill(0); y.forEach(c => counts[c]++);
    const maxC = Math.max(...counts), idx = [];
    for (let c = 0; c < nClasses; c++) {
      const own = y.map((v, i) => v === c ? i : -1).filter(i => i >= 0);
      for (let k = 0; k < maxC; k++) idx.push(own[k % own.length]);
    }
    const D = X[0].length;
    const Xt = tf.tensor2d(idx.map(i => X[i])), Yt = tf.oneHot(tf.tensor1d(idx.map(i => y[i]), "int32"), nClasses).toFloat();
    const W = tf.variable(tf.randomNormal([D, nClasses], 0, 0.01)), b = tf.variable(tf.zeros([nClasses]));
    const opt = tf.train.adam(lr);
    for (let e = 0; e < epochs; e++) {
      const loss = opt.minimize(() => tf.losses.softmaxCrossEntropy(Yt, Xt.matMul(W).add(b)).add(W.square().sum().mul(l2)), true);
      if (onEpoch && (e % 20 === 0 || e === epochs - 1)) { onEpoch(e + 1, (await loss.data())[0]); await tf.nextFrame(); }
      loss.dispose();
    }
    const out = { W: await W.array(), b: Array.from(await b.data()) };
    [Xt, Yt, W, b].forEach(t => t.dispose());
    return out;
  }

  return { SIZE, loadBase, toInput, embed, makeClassifier, train };
})();
