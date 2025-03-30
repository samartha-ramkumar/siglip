import os
import pandas as pd
import keras_cv
import tensorflow as tf
from src_tf.utils.config import CFG
import keras_nlp

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    preset=CFG.text_preset, # Name of the model
    sequence_length=CFG.sequence_length, # Max sequence length, will be padded if shorter
)


def build_augmenter():
    # Define augmentations
    aug_layers = [
        keras_cv.layers.RandomBrightness(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomContrast(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomSaturation(factor=(0.45, 0.55)),
        keras_cv.layers.RandomHue(factor=0.1, value_range=(0, 1)),
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.15), width_factor=(0.06, 0.15)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomZoom(height_factor=(0.05, 0.10)),
        keras_cv.layers.RandomRotation(factor=(0.01, 0.05)),
    ]
    
    # Apply augmentations to random samples
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    
    # Build augmentation layer
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    # Apply augmentations
    def augment(inp):
        inp["images"] = augmenter({"images": inp["images"]})["images"]
        return inp
    return augment


def build_decoder(target_size=CFG.image_size):
    def decode_image(image_path):
        # Read jpeg image
        file_bytes = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(file_bytes)
        
        # Resize
        image = tf.image.resize(image, size=target_size, method="area")
        
        # Rescale image
        image = tf.cast(image, tf.float32)
        image /= 255.0
        
        # Reshape
        image = tf.reshape(image, [*target_size, 3])
        return image

    def decode_text(text):
        text = preprocessor(text)
        return text

    def decode_input(image_path, text):
        image = decode_image(image_path)
        text = decode_text(text)
        return {"images":image, "texts":text}

    return decode_input


def build_dataset(
    image_paths,
    texts,
    batch_size=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    drop_remainder=True,
):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder()

    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.experimental.AUTOTUNE

    slices = (image_paths, texts)
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds


def load_dataset():
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

    # train_df = sample_df[sample_df.fold != 0]  # Uncomment this line for dataset training
    train_df = sample_df[sample_df.fold == 1]
    print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

    # Train
    train_paths = train_df.image_path.values
    train_texts = train_df.caption.values
    train_ds = build_dataset(train_paths, train_texts,
                             batch_size=CFG.batch_size,
                             repeat=True, shuffle=True, augment=True, cache=True)

    # Valid
    valid_paths = valid_df.image_path.values
    valid_texts = valid_df.caption.values
    valid_ds = build_dataset(valid_paths, valid_texts,
                            batch_size=CFG.batch_size,
                            repeat=False, shuffle=False, augment=False, cache=True)
    
    return train_ds, valid_ds