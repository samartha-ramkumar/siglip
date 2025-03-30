import matplotlib as plt
import os
import tensorflow as tf
import keras_nlp
from src_tf.utils.config import CFG


preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    preset=CFG.text_preset, # Name of the model
    sequence_length=CFG.sequence_length, # Max sequence length, will be padded if shorter
)

def debug_dataloader(df):
    my_df = df.groupby("image").head(1).reset_index(drop=True) # .sample(frac=1.0)
    my_df = my_df[my_df.fold == 0]
    my_df = my_df.head(15)

    my_paths = my_df.image_path.values
    my_texts = my_df.caption.values



    def build_my_dataset(
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
        print(ds)
        print(list(ds.as_numpy_iterator()))
        print(type(ds))
        print(list(ds.as_numpy_iterator()))
        print(type(ds))
        ds = ds.map(decode_fn, num_parallel_calls=AUTO)
        samples = list(ds.as_numpy_iterator())
        print("==============================")
        print(samples[0]['images'].size)
        print(samples[0]['images'].shape)
        print(type(samples[0]['images']))
        ds = ds.cache(cache_dir) if cache else ds
        ds = ds.repeat() if repeat else ds

        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        print("==============================")
        print(ds)
        print(type(ds))
        samples1 = list(ds.as_numpy_iterator())
        print(samples1[0]['images'].shape)
        ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
        ds = ds.prefetch(AUTO)
        return ds

    my_ds = build_my_dataset(my_paths, my_texts,
                            batch_size=10,
                            repeat=False, shuffle=False, augment=False, cache=True)

    print("==================================")
    for i,s in my_ds.enumerate():
        print(i)
        print(s.keys())
        break



def visualize(valid_ds):
    batch = next(iter(valid_ds))
    imgs = batch["images"]
    txts = batch["texts"]

    fig = plt.figure(figsize=(15, 10)) 
    for i in range(6):
        img = imgs[i].numpy()
        print(preprocessor.tokenizer.detokenize(txts["token_ids"][i]))
        caption = preprocessor.tokenizer.detokenize(txts["token_ids"][i])
        print(caption)
        caption = caption.replace("[PAD]","").replace("[CLS]","").replace("[SEP]","").strip()
        caption = " ".join(caption.split(" ")[:12]) + "\n" + " ".join(caption.split(" ")[12:])
        plt.subplot(2, 3, i + 1) 
        plt.imshow(img)
        plt.axis('off')
        plt.title(caption, fontsize=12) 

    plt.tight_layout()
    plt.show()