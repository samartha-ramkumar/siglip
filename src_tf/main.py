from src_tf.utils.config import CFG
import keras
import pandas as pd
from src_tf.data.loader import build_dataset

keras.utils.set_random_seed(CFG.seed)
df = pd.read_csv(f"{CFG.caption_path}/captions.txt")
df["image_path"] = CFG.image_path + "/" + df.image


from sklearn.model_selection import GroupKFold

# Create a GroupKFold object with 5 folds
gkf = GroupKFold(n_splits=5)

# Add fold column based on groups
df['fold'] = -1
for fold, (train_index, valid_index) in enumerate(gkf.split(df, groups=df["image"])):
    df.loc[valid_index, 'fold'] = fold



sample_df = df.groupby("image").head(1).reset_index(drop=True) # .sample(frac=1.0)
valid_df = sample_df[sample_df.fold == 0]

# train_df = sample_df[sample_df.fold != 0]
# print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# Train
# train_paths = train_df.image_path.values
# train_texts = train_df.caption.values
# train_ds = build_dataset(train_paths, train_texts,
#                          batch_size=CFG.batch_size,
#                          repeat=True, shuffle=True, augment=True, cache=True)

# Valid
valid_paths = valid_df.image_path.values
valid_texts = valid_df.caption.values
valid_ds = build_dataset(valid_paths, valid_texts,
                         batch_size=CFG.batch_size,
                         repeat=False, shuffle=False, augment=False, cache=True)


from src_tf.models.encoders.encoder import build_image_encoder,  build_text_encoder
from src_tf.utils.loss import SigLIPLoss

image_encoder = build_image_encoder()
text_encoder = build_text_encoder()

image_encoder = build_image_encoder()
text_encoder = build_text_encoder()
# image_encoder.summary(), text_encoder.summary()


from src_tf.utils.lr_scheduler import get_lr_callback

lr_cb = get_lr_callback(CFG.batch_size, mode="cos", plot=True)