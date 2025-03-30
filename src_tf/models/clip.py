from siglip.src_tf.utils.config import CFG
import keras
from keras import ops


class CLIPModel(keras.Model):
    def __init__(self, image_encoder, text_encoder, temperature=1.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

    def compile(
        self,
        optimizer,
        loss,
    ):
        super().compile(optimizer=optimizer)
        self.loss = loss

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        image_features, text_features = self.get_features(x, training=True)
        logits = self.get_logits(image_features, text_features)
        labels = self.get_ground_truth(image_features, text_features)
        return self.loss(labels, logits)

    def get_features(self, x, training):
        image_features = self.image_encoder(x["images"], training=training)
        text_features = self.text_encoder(x["texts"], training=training)
        return image_features, text_features

    def get_ground_truth(self, image_features, text_features):
        image_scores = image_features @ ops.transpose(image_features)
        text_scores = text_features @ ops.transpose(text_features)
        labels = ops.softmax(
            (image_scores + text_scores) / 2 * self.temperature, axis=-1
        )
        labels = ops.cast(labels, dtype="float32")
        return labels

    def get_logits(self, image_features, text_features, logit_scale=1.0):
        logits = image_features @ ops.transpose(text_features) / self.temperature
        logits = ops.cast(logits, dtype="float32")
        return logits

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, x, training=False):
        image_features, text_features = self.get_features(x, training=training)
        logits = self.get_logits(image_features, text_features)
        return logits
    
if __name__ == '__main__':
    from siglip.src_tf.models.encoders.encoder import build_image_encoder,  build_text_encoder
    from siglip.src_tf.utils.loss import CLIPLoss

    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()
    image_encoder.summary(), text_encoder.summary()
    clip_model = CLIPModel(image_encoder, text_encoder)

    clip_model.compile(optimizer="adam", loss=CLIPLoss())