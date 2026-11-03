"""Milestone 4: IRv2 baseline vs IRv2 + focal loss on HAM10000.

Script port of cv_project_irv2.ipynb (the M2 Kaggle run). The pipeline is unchanged: same split,
offline augmentation, backbone cut, head, optimiser, callbacks and evaluation. The only new option
is --loss, which swaps categorical cross-entropy for categorical focal loss (gamma=2, alpha=1),
the single variable of the M3 hypothesis (M3_hypothesis.md).

    python train.py --loss ce    --seed 42
    python train.py --loss focal --seed 7
    QUICK_CHECK=1 python train.py --loss focal --seed 42   # smoke test: tiny subset, 1 epoch
"""
import os, argparse
import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--model", choices=["irv2", "irv2_sa"], default="irv2")
ap.add_argument("--protocol", choices=["paper", "clean"], default="paper")
ap.add_argument("--data-dir", default=None, help="folder containing HAM10000_metadata.csv")
ap.add_argument("--work-dir", default=None, help="where runs/ is written (default: ./work)")
ap.add_argument("--aug-dir", default=None, help="where the ~1.3 GB augmented set goes (use fast local disk)")
args = ap.parse_args()

MODEL     = args.model      # "irv2" = baseline (no attention) | "irv2_sa" = with Soft-Attention
PROTOCOL  = args.protocol   # "paper" = exact authors' protocol (test set doubles as validation set)
                            # "clean" = lesion-grouped validation split carved from train; test touched once
SEED      = args.seed
LOSS      = args.loss       # "ce" = categorical cross-entropy (paper) | "focal" = M3 hypothesis
FOCAL_GAMMA, FOCAL_ALPHA = 2.0, 1.0   # alpha=1: no re-weighting on top of the mel x5 class weight

# --- hyper-parameters exactly as in the official notebooks ---
IMG_SIZE      = 299     # IRv2 input
BATCH_SIZE    = 16
EPOCHS        = 150
PATIENCE      = 30      # EarlyStopping(monitor="val_loss", patience=30, min_delta=0.001)
LR, ADAM_EPS  = 0.01, 0.1
TEST_SIZE     = 0.15    # 85/15 split
AUG_TARGET    = 8000    # offline augmentation target per class
AUG_BATCH     = 50      # generator batch size used by the authors (affects exact image counts)
CLASS_WEIGHTS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 5.0, 5: 1.0, 6: 1.0}  # mel x5 (in code, not in paper)

MIXED_PRECISION = False  # kept off: it would be a second change next to the loss
DATA_DIR        = args.data_dir

# QUICK_CHECK=1 runs a tiny smoke test (few images, 1 epoch, no ImageNet download) to verify the pipeline
QUICK_CHECK = os.environ.get("QUICK_CHECK", "0") == "1"
if QUICK_CHECK:
    AUG_TARGET, AUG_BATCH, EPOCHS, PATIENCE = 60, 10, 1, 1

CLASSES  = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # alphabetical = Keras flow_from_directory order
RUN_NAME = f"{MODEL}_{LOSS}_{PROTOCOL}_seed{SEED}" + ("_quick" if QUICK_CHECK else "")
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK_DIR = args.work_dir or ("/kaggle/working" if ON_KAGGLE else os.path.abspath("work"))
OUT_DIR  = os.path.join(WORK_DIR, "runs", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
print("Run:", RUN_NAME, "| outputs ->", OUT_DIR)


# ------------------------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------------------------
import json, math, glob, time, random, shutil, platform
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, roc_curve, balanced_accuracy_score)
from concurrent.futures import ThreadPoolExecutor

keras.utils.set_random_seed(SEED)          # seeds python, numpy and TF
gpus = tf.config.list_physical_devices("GPU")
print("Python", platform.python_version(), "| TF", tf.__version__, "| Keras", keras.__version__)
print("GPUs:", gpus or "NONE (training on CPU is not feasible for the full run)")
if MIXED_PRECISION:
    keras.mixed_precision.set_global_policy("mixed_float16")

# ------------------------------------------------------------------------------------------------
# Data: locate and verify HAM10000
# ------------------------------------------------------------------------------------------------
def find_data_dir():
    if DATA_DIR:
        return DATA_DIR
    for root in ["/kaggle/input", "/workspace", "/content", os.getcwd()]:
        hits = glob.glob(os.path.join(root, "**", "HAM10000_metadata.csv"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    import kagglehub                       # Colab / local fallback (public dataset)
    return kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

data_dir = find_data_dir()
meta = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))

# image paths (the Kaggle mirror ships part_1 / part_2, sometimes twice in different case -> de-duplicate by id)
img_path = {}
for p in glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True):
    img_path.setdefault(os.path.splitext(os.path.basename(p))[0], p)
meta = meta[meta["image_id"].isin(img_path)].reset_index(drop=True)
meta["path"] = meta["image_id"].map(img_path)

if QUICK_CHECK:  # tiny stratified subset that keeps multi-image lesions
    meta = meta.groupby("dx").head(30).reset_index(drop=True)

print("data dir:", data_dir)
print(f"images: {len(meta):,} | distinct lesions: {meta.lesion_id.nunique():,}")
dist = meta["dx"].value_counts().reindex(CLASSES).to_frame("images")
dist["%"] = (100 * dist["images"] / dist["images"].sum()).round(1)
print(dist)

if not QUICK_CHECK:
    assert len(meta) == 10015, "expected 10,015 images"
    assert meta.lesion_id.nunique() == 7470, "expected 7,470 lesions"
    print("Dataset matches the HAM10000 descriptor (10,015 images / 7,470 lesions).")

# ------------------------------------------------------------------------------------------------
# Train/test split (the authors' exact logic)
# ------------------------------------------------------------------------------------------------
n_img = meta.groupby("lesion_id")["image_id"].transform("count")
single = meta[n_img == 1]                                         # test candidate pool
_, test_df = train_test_split(single, test_size=TEST_SIZE, stratify=single["dx"], random_state=SEED)
train_df = meta[~meta["image_id"].isin(test_df["image_id"])]

if PROTOCOL == "clean":
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    tr_idx, va_idx = next(gss.split(train_df, groups=train_df["lesion_id"]))
    val_df, train_df = train_df.iloc[va_idx], train_df.iloc[tr_idx]
    assert not set(val_df.lesion_id) & set(train_df.lesion_id)
else:
    val_df = test_df                                              # authors' protocol

assert not set(test_df.lesion_id) & set(train_df.lesion_id), "lesion leakage!"
print(f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}   (paper: train 9,187 | test 828)")

paper_support = {"akiec": 23, "bcc": 26, "bkl": 66, "df": 6, "mel": 34, "nv": 663, "vasc": 10}
split_tbl = pd.DataFrame({
    "train (before aug)": train_df.dx.value_counts().reindex(CLASSES),
    "test (ours)": test_df.dx.value_counts().reindex(CLASSES),
    "test (paper)": pd.Series(paper_support),
})
print(split_tbl)

# ------------------------------------------------------------------------------------------------
# Offline augmentation (after the split)
# ------------------------------------------------------------------------------------------------
def n_augmented(n, target=AUG_TARGET, bs=AUG_BATCH):
    # images produced by the authors' loop: num_batches batches cycling over n files in chunks of bs
    num_batches = max(0, math.ceil((target - n) / bs))
    pattern = [bs] * (n // bs) + ([n % bs] if n % bs else [])
    return sum(pattern[i % len(pattern)] for i in range(num_batches))

def random_affine(img, rng):
    # ImageDataGenerator(rotation_range=180, width/height_shift_range=0.1, zoom_range=0.1,
    #                    horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    h, w = img.shape[:2]
    theta = np.deg2rad(rng.uniform(-180, 180))
    tx, ty = rng.uniform(-0.1, 0.1) * h, rng.uniform(-0.1, 0.1) * w
    zx, zy = rng.uniform(0.9, 1.1, size=2)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    Z = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    M = R @ T @ Z
    o = np.array([[1, 0, h / 2 - 0.5], [0, 1, w / 2 - 0.5], [0, 0, 1]])          # transform about centre
    M = o @ M @ np.linalg.inv(o)
    # Keras matrices act on (row, col); convert to OpenCV's (x=col, y=row) convention
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    M_cv = P @ M @ P
    out = cv2.warpAffine(img, M_cv[:2], (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5: out = out[:, ::-1]
    if rng.random() < 0.5: out = out[::-1, :]
    return out

def load_resized(path):
    img = cv2.imread(path)                                         # BGR, kept as BGR for writing
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

# on Kaggle keep the ~1.3 GB augmented set in /tmp so it is not saved as notebook output
AUG_DIR = os.path.join(args.aug_dir or ("/tmp" if ON_KAGGLE else WORK_DIR), "aug", f"{PROTOCOL}_seed{SEED}" + ("_quick" if QUICK_CHECK else ""))
manifest_path = os.path.join(AUG_DIR, "manifest.csv")

if os.path.exists(manifest_path):
    aug_df = pd.read_csv(manifest_path)
    print("re-using augmented set:", AUG_DIR)
else:
    t0, rows = time.time(), []
    for ci, c in enumerate(CLASSES):
        os.makedirs(os.path.join(AUG_DIR, c), exist_ok=True)
        src = train_df.loc[train_df.dx == c, "path"].tolist()
        k = n_augmented(len(src))
        order = []                                                  # cycle through the class like the generator
        rng_c = np.random.default_rng(SEED + ci)
        while len(order) < k:
            order += list(rng_c.permutation(len(src)))
        order = order[:k]

        def make(j, c=c, src=src, ci=ci, order=order):
            rng = np.random.default_rng([SEED, ci, j])
            out = random_affine(load_resized(src[order[j]]), rng)
            dst = os.path.join(AUG_DIR, c, f"aug_{j:05d}.jpg")
            cv2.imwrite(dst, out)
            return dst
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            dsts = list(ex.map(make, range(k)))
        rows += [(p, c) for p in src] + [(p, c) for p in dsts]
        print(f"{c:>6}: {len(src):5d} originals + {k:5d} augmented = {len(src) + k:5d}")
    aug_df = pd.DataFrame(rows, columns=["path", "dx"])
    aug_df.to_csv(manifest_path, index=False)
    print(f"augmentation took {time.time() - t0:.0f}s")

print(f"training images after augmentation: {len(aug_df):,}   (authors' notebook: 51,699)")

# sanity check: a few augmented samples
fig, axs = plt.subplots(1, 7, figsize=(16, 2.6))
for ax, c in zip(axs, CLASSES):
    p = aug_df[(aug_df.dx == c) & aug_df.path.str.contains("aug_")].path.iloc[0]
    ax.imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)); ax.set_title(c); ax.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "augmented_samples.png"), dpi=100); plt.close()

# ------------------------------------------------------------------------------------------------
# Input pipeline
# ------------------------------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
preprocess = keras.applications.inception_resnet_v2.preprocess_input
cls_index = {c: i for i, c in enumerate(CLASSES)}

def decode(path, label):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="nearest")
    img = preprocess(tf.cast(img, tf.float32))
    return img, tf.one_hot(label, len(CLASSES))

def make_ds(df, train):
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["dx"].map(cls_index).values))
    if train:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True).repeat()
    ds = ds.map(decode, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)

# keep test order sorted by class (like flow_from_directory) so outputs are easy to read
test_df = test_df.assign(_c=test_df.dx.map(cls_index)).sort_values(["_c", "image_id"]).drop(columns="_c")
val_df = test_df if PROTOCOL == "paper" else val_df
train_ds, val_ds, test_ds = make_ds(aug_df, True), make_ds(val_df, False), make_ds(test_df, False)
STEPS_PER_EPOCH = math.ceil(len(train_df) / 10)  # authors: steps_per_epoch = len(train_df)/10  -> 919
print("steps/epoch:", STEPS_PER_EPOCH, "| images seen per epoch:", STEPS_PER_EPOCH * BATCH_SIZE)

# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class SoftAttention(layers.Layer):
    # Port of the authors' SoftAttention (aggregate / concat options kept)
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        super().__init__(**kwargs)
        self.channels, self.multiheads = int(ch), m
        self.aggregate_channels, self.concat_input_with_scaled = aggregate, concat_with_x

    def build(self, input_shape):
        self.i_shape = tuple(input_shape)
        self.kernel_conv3d = self.add_weight(shape=(self.channels, 3, 3, 1, self.multiheads),
                                             initializer="he_uniform", name="kernel_conv3d")
        self.bias_conv3d = self.add_weight(shape=(self.multiheads,), initializer="zeros", name="bias_conv3d")

    def call(self, x):
        h, w, c = self.i_shape[1], self.i_shape[2], self.i_shape[3]
        exp_x = tf.expand_dims(x, -1)                                           # (B,H,W,C,1)
        k = tf.cast(self.kernel_conv3d, x.dtype)
        c3d = tf.nn.conv3d(exp_x, k, strides=[1, 1, 1, c, 1], padding="SAME")  # (B,H,W,1,m)
        c3d = tf.nn.relu(tf.nn.bias_add(c3d, tf.cast(self.bias_conv3d, x.dtype)))
        c3d = tf.squeeze(tf.transpose(c3d, (0, 4, 1, 2, 3)), -1)               # (B,m,H,W)
        alpha = tf.nn.softmax(tf.reshape(c3d, (-1, self.multiheads, h * w)), axis=-1)
        alpha = tf.reshape(alpha, (-1, self.multiheads, h, w))
        if not self.aggregate_channels:
            a = tf.transpose(tf.expand_dims(alpha, -1), (0, 2, 3, 1, 4))         # (B,H,W,m,1)
            u = a * tf.expand_dims(x, -2)                                       # (B,H,W,m,C)
            u = tf.reshape(u, (-1, h, w, self.multiheads * c))
        else:
            a = tf.reduce_sum(tf.transpose(alpha, (0, 2, 3, 1)), -1, keepdims=True)  # (B,H,W,1)
            u = a * x
        o = tf.concat([u, x], -1) if self.concat_input_with_scaled else u
        return [o, alpha]

    def compute_output_shape(self, input_shape):
        h, w, c = input_shape[1:]
        if not self.aggregate_channels:
            out_c = c + c * self.multiheads
        else:
            out_c = 2 * c if self.concat_input_with_scaled else c
        return [(input_shape[0], h, w, out_c), (input_shape[0], self.multiheads, h, w)]

    def get_config(self):
        return {**super().get_config(), "ch": self.channels, "m": self.multiheads,
                "concat_with_x": self.concat_input_with_scaled, "aggregate": self.aggregate_channels}


def build_model(kind):
    irv2 = keras.applications.InceptionResNetV2(include_top=True, classifier_activation="softmax",
                                                weights=None if QUICK_CHECK else "imagenet")
    names = [l.name for l in irv2.layers]
    # authors' cut point: BN after the first 1x1 conv (192 ch) that consumes block8_8_ac
    cut = irv2.layers[names.index("block8_8_ac") + 2]
    assert isinstance(cut, layers.BatchNormalization) and tuple(cut.output.shape[1:]) == (8, 8, 192), cut.name
    build_model.cut_name = cut.name
    conv = cut.output
    if kind == "irv2_sa":
        att, _ = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=int(conv.shape[-1]),
                               name="soft_attention")(conv)
        att = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(att)
        conv = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(conv)
        conv = layers.concatenate([conv, att])
    conv = layers.Activation("relu")(conv)
    conv = layers.Dropout(0.5)(conv)
    out = layers.Flatten()(conv)
    out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32")(out)
    return keras.Model(irv2.input, out, name=kind)

model = build_model(MODEL)
expected = {"irv2": 47_550_631, "irv2_sa": 47_535_287}[MODEL]            # from the authors' model.summary()
print(f"cut layer: {build_model.cut_name} | params: {model.count_params():,} "
      f"(authors: {expected:,}) -> {'MATCH' if model.count_params() == expected else 'MISMATCH'}")

# ------------------------------------------------------------------------------------------------
# Training (authors' settings; only the loss can change)
# ------------------------------------------------------------------------------------------------
# FL(p_t) = -alpha (1 - p_t)^gamma log(p_t); gamma = 0 gives cross-entropy, so the loss is the only change
loss_fn = ("categorical_crossentropy" if LOSS == "ce" else
           keras.losses.CategoricalFocalCrossentropy(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR, epsilon=ADAM_EPS),
              loss=loss_fn, metrics=["accuracy"])

class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.t = time.time()
    def on_epoch_end(self, epoch, logs=None): logs["epoch_time_s"] = time.time() - self.t

ckpt_path = os.path.join(OUT_DIR, "best.weights.h5")
callbacks = [
    EpochTimer(),
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, min_delta=0.001),
    keras.callbacks.BackupAndRestore(os.path.join(OUT_DIR, "backup")),    # resume after a Colab disconnect
    keras.callbacks.CSVLogger(os.path.join(OUT_DIR, "history.csv"), append=True),
]

t0 = time.time()
history = model.fit(train_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=val_ds,
                    class_weight=CLASS_WEIGHTS, callbacks=callbacks, verbose=2)
train_time_h = (time.time() - t0) / 3600
print(f"training wall-clock: {train_time_h:.2f} h on {gpus[0].name if gpus else 'CPU'}")

hist = pd.read_csv(os.path.join(OUT_DIR, "history.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.epoch, hist.loss, label="train"); ax[0].plot(hist.epoch, hist.val_loss, label="val")
ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist.epoch, hist.accuracy, label="train"); ax[1].plot(hist.epoch, hist.val_accuracy, label="val")
ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=120); plt.close()
best_ep = int(hist.val_accuracy.idxmax())
print(f"epochs run: {len(hist)} | best val_accuracy at epoch {best_ep + 1} | "
      f"mean epoch time: {hist.epoch_time_s.mean():.0f}s")

# ------------------------------------------------------------------------------------------------
# Evaluation on the held-out test set
# ------------------------------------------------------------------------------------------------
model.load_weights(ckpt_path)
y_prob = model.predict(test_ds, verbose=0)
y_true = test_df["dx"].map(cls_index).values
y_pred = y_prob.argmax(1)
Y = np.eye(len(CLASSES))[y_true]

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0))

def specificity(y_t, y_p, k):
    neg = y_t != k
    return float(((y_p != k) & neg).sum() / neg.sum())

summary = {
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy (=macro recall)": balanced_accuracy_score(y_true, y_pred),
    "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
    "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "auc_weighted_ovr": roc_auc_score(Y, y_prob, average="weighted", multi_class="ovr"),
    "auc_macro_ovr": roc_auc_score(Y, y_prob, average="macro", multi_class="ovr"),
}
per_class = pd.DataFrame({
    "support": np.bincount(y_true, minlength=7),
    "precision": precision_score(y_true, y_pred, average=None, labels=range(7), zero_division=0),
    "recall (sensitivity)": recall_score(y_true, y_pred, average=None, labels=range(7), zero_division=0),
    "specificity": [specificity(y_true, y_pred, k) for k in range(7)],
    "f1": f1_score(y_true, y_pred, average=None, labels=range(7), zero_division=0),
    "auc": [roc_auc_score(Y[:, k], y_prob[:, k]) if 0 < Y[:, k].sum() < len(Y) else np.nan for k in range(7)],
}, index=CLASSES).round(3)
print(pd.Series(summary).round(4).to_frame("ours"))
print(per_class)

# ------------------------------------------------------------------------------------------------
# Comparison with the paper
# ------------------------------------------------------------------------------------------------
REF = {  # authors' notebook outputs (consistent with paper Table 1 / Table 3)
    "irv2": {
        "precision": [0.83, 0.85, 0.85, 0.67, 0.70, 0.93, 1.00], "recall": [0.65, 0.85, 0.44, 0.33, 0.56, 0.99, 1.00],
        "auc": [0.993, 0.997, 0.971, 0.983, 0.966, 0.984, 1.000],
        "summary": {"accuracy": 0.912, "precision_weighted": 0.906, "precision_macro": 0.833,
                    "recall_macro": 0.689, "auc_weighted_ovr": 0.983, "auc_macro_ovr": 0.985}},
    "irv2_sa": {
        "precision": [1.00, 0.88, 0.72, 1.00, 0.67, 0.97, 1.00], "recall": [0.52, 0.88, 0.83, 0.17, 0.65, 0.98, 1.00],
        "auc": [0.981, 0.999, 0.982, 0.974, 0.974, 0.984, 1.000],
        "summary": {"accuracy": 0.935, "precision_weighted": 0.938, "precision_macro": 0.892,
                    "recall_macro": 0.719, "auc_weighted_ovr": 0.984, "auc_macro_ovr": 0.985}},
}[MODEL]

cmp_summary = pd.DataFrame({"paper": pd.Series(REF["summary"]),
                            "ours": pd.Series(summary).reindex(REF["summary"].keys())})
cmp_summary["gap (ours - paper)"] = cmp_summary["ours"] - cmp_summary["paper"]
print(cmp_summary.round(3))

cmp_class = pd.DataFrame({
    "prec paper": REF["precision"], "prec ours": per_class["precision"].values,
    "recall paper": REF["recall"], "recall ours": per_class["recall (sensitivity)"].values,
    "auc paper": REF["auc"], "auc ours": per_class["auc"].values,
    "support ours": per_class["support"].values,
}, index=CLASSES)
print(cmp_class.round(3))

# 95% bootstrap confidence intervals: the test set has only ~6 df / ~10 vasc images
rng = np.random.default_rng(SEED)
boot = {"precision_weighted": [], "recall_macro": [], "mel_recall": [], "auc_weighted_ovr": []}
for _ in range(1000):
    i = rng.integers(0, len(y_true), len(y_true))
    yt, yp, pr = y_true[i], y_pred[i], y_prob[i]
    boot["precision_weighted"].append(precision_score(yt, yp, average="weighted", zero_division=0))
    boot["recall_macro"].append(recall_score(yt, yp, average="macro", zero_division=0))
    boot["mel_recall"].append(recall_score(yt == 4, yp == 4, zero_division=0))
    if len(np.unique(yt)) == 7:
        boot["auc_weighted_ovr"].append(roc_auc_score(np.eye(7)[yt], pr, average="weighted", multi_class="ovr"))
ci = pd.DataFrame({k: np.percentile(v, [2.5, 50, 97.5]) for k, v in boot.items() if v},
                  index=["2.5%", "median", "97.5%"]).T.round(3)
print(ci)

# ------------------------------------------------------------------------------------------------
# Confusion matrix and ROC curves
# ------------------------------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=range(7))
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
for a, m, t, fmt in [(ax[0], cm, "counts", "d"), (ax[1], cmn, "row-normalised (recall)", ".2f")]:
    im = a.imshow(m, cmap="Blues")
    a.set_xticks(range(7), CLASSES); a.set_yticks(range(7), CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(f"{MODEL} — {t}")
    for r in range(7):
        for c in range(7):
            a.text(c, r, format(m[r, c], fmt), ha="center", va="center",
                   color="white" if m[r, c] > m.max() / 2 else "black", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=120); plt.close()

plt.figure(figsize=(6, 5.5))
for k, c in enumerate(CLASSES):
    if 0 < Y[:, k].sum() < len(Y):
        fpr, tpr, _ = roc_curve(Y[:, k], y_prob[:, k])
        plt.plot(fpr, tpr, label=f"{c} (AUC {per_class.loc[c, 'auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8); plt.xlabel("FPR"); plt.ylabel("TPR (sensitivity)")
plt.title(f"One-vs-rest ROC — {MODEL}"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "roc.png"), dpi=120); plt.close()

# Most frequent confusions (input for the Stage-6 failure analysis)
off = [(CLASSES[r], CLASSES[c], int(cm[r, c])) for r in range(7) for c in range(7) if r != c and cm[r, c]]
print(pd.DataFrame(sorted(off, key=lambda x: -x[2])[:10], columns=["true", "predicted", "count"]))

# A few misclassified test images, most confident mistakes first
wrong = np.where(y_pred != y_true)[0]
wrong = wrong[np.argsort(-y_prob[wrong].max(1))][:10]
if len(wrong):
    fig, axs = plt.subplots(2, 5, figsize=(16, 7))
    for a in axs.ravel(): a.axis("off")
    for a, i in zip(axs.ravel(), wrong):
        a.imshow(cv2.cvtColor(cv2.imread(test_df.path.iloc[i]), cv2.COLOR_BGR2RGB))
        a.set_title(f"true {CLASSES[y_true[i]]} -> pred {CLASSES[y_pred[i]]}\np={y_prob[i].max():.2f}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "failures.png"), dpi=110); plt.close()

# ------------------------------------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------------------------------------
pd.DataFrame(y_prob, columns=[f"p_{c}" for c in CLASSES]).assign(
    image_id=test_df.image_id.values, true=test_df.dx.values, pred=[CLASSES[k] for k in y_pred]
).to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
per_class.to_csv(os.path.join(OUT_DIR, "per_class.csv"))
cmp_summary.to_csv(os.path.join(OUT_DIR, "comparison_with_paper.csv"))
results = {
    "run": RUN_NAME, "model": MODEL, "loss": LOSS, "protocol": PROTOCOL, "seed": SEED, "quick_check": QUICK_CHECK,
    "tf": tf.__version__, "keras": keras.__version__, "gpu": gpus[0].name if gpus else "CPU",
    "n_train_before_aug": len(train_df), "n_train_after_aug": len(aug_df), "n_val": len(val_df), "n_test": len(test_df),
    "epochs_run": len(hist), "best_epoch": best_ep + 1, "train_time_h": round(train_time_h, 2),
    "params": model.count_params(), "summary": {k: round(float(v), 4) for k, v in summary.items()},
    "per_class": per_class.to_dict(), "bootstrap_ci": ci.to_dict(),
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved to", OUT_DIR); print(sorted(os.listdir(OUT_DIR)))
