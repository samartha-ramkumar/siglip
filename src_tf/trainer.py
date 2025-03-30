"""
Training and inference
"""

import cv2
import keras
import keras_nlp
import numpy as np
from keras import ops
import pandas as pd
import matplotlib as plt
from src_tf.utils.config import CFG
from src_tf.data.loader import load_dataset

from src_tf.models.encoders.encoder import build_image_encoder,  build_text_encoder
from src_tf.utils.loss import SigLIPLoss
from src_tf.utils.lr_scheduler import get_lr_callback
from src_tf.models.siglip import SigLIPModel



keras.utils.set_random_seed(CFG.seed)

train_ds, valid_ds = load_dataset()


def train():
    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()

    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()
    # print(image_encoder.summary(), text_encoder.summary())


    lr_cb = get_lr_callback(CFG.batch_size, mode="cos", plot=True)

    ckpt_cb = keras.callbacks.ModelCheckpoint("best_model.keras",
                                            monitor='val_loss',
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='min')

    siglip_model = SigLIPModel(
                            image_encoder, text_encoder, 
                            num_logits=CFG.batch_size,
                            logit_scale=2.30, logit_bias=-10.0)

    siglip_model.compile(optimizer="adam", loss=SigLIPLoss())

    history = siglip_model.fit(
        train_ds,
        epochs=CFG.epochs,
        callbacks=[lr_cb, ckpt_cb],
        steps_per_epoch= 8090 // CFG.batch_size,
        validation_data=valid_ds,
        verbose=1,
    )
    
    # print(f"history : {history}")




def inference():
    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
                            preset=CFG.text_preset, # Name of the model
                            sequence_length=CFG.sequence_length, # Max sequence length, will be padded if shorter
                            )

    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()

    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()
    # image_encoder.summary(), text_encoder.summary()

    siglip_model = SigLIPModel(
                        image_encoder, text_encoder, 
                        num_logits=CFG.batch_size,
                        logit_scale=2.30, logit_bias=-10.0)

    siglip_model.compile(optimizer="adam", loss=SigLIPLoss())
    siglip_model.load_weights("./data/best_model.keras")


    def process_image(path):
        img = cv2.imread(path)[...,::-1] # BGR -> RGB
        img = cv2.resize(img, dsize=CFG.image_size, interpolation=cv2.INTER_AREA)
        img = img / 255.0 
        return img

    def process_text(text):
        text = [f"a photo of a {x}" for x in text]
        return preprocessor(text)

    def zero_shot_classifier(image_path, candid_labels):
        image = process_image(image_path)
        plt.imshow(image)
        image = ops.convert_to_tensor(image)[None,]
        text = process_text(candid_labels)
        pred = siglip_model({"images":image, "texts":text})
        pred = ops.softmax(pred)*100
        pred = ops.convert_to_numpy(pred).tolist()[0]
        pred = dict(zip(candid_labels, np.round(pred, 2)))
        plt.title(f"Prediction: {pred}", fontsize=10)
        plt.show()
        return pred

    # Download sample data independently
    # wget "https://img.freepik.com/free-psd/banana-character-isolated_23-2151170924.jpg" -O "banana.jpg"

    pred = zero_shot_classifier(image_path="banana.jpg",
                                candid_labels=["banana", "man", "dog"])
    
    print(f"prediction : {pred}")