from src_tf.utils.config import CFG
import keras
from keras import ops

class SigLIPModel(keras.Model):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        num_logits,
        logit_scale,
        logit_bias,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.num_logits = num_logits
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias

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
        logits = self.get_logits(x, training=True)
        labels = self.get_ground_truth(self.num_logits)
        return self.loss(labels, logits)

    def get_ground_truth(self, num_logits):
        labels = -ops.ones((num_logits, num_logits))
        labels = labels + 2 * ops.eye(num_logits)
        labels = ops.cast(labels, dtype="float32")
        return labels

    def get_logits(self, x, training):
        image_features = self.image_encoder(x["images"], training=training)
        text_features = self.text_encoder(x["texts"], training=training)
        logits = image_features @ ops.transpose(text_features)
        logits = self.logit_scale * logits + self.logit_bias
        logits = ops.cast(logits, dtype="float32")
        return logits

    def build(self, input_shape):
        super().build(input_shape)
        self.built = True

    def call(self, x, training=False):
        return self.get_logits(x, training=training)


if __name__ == '__main__':
    from src_tf.models.encoders.encoder import build_image_encoder,  build_text_encoder
    from src_tf.utils.loss import SigLIPLoss
    
    image_encoder = build_image_encoder()
    text_encoder = build_text_encoder()
    image_encoder.summary(), text_encoder.summary()

    siglip_model = SigLIPModel(image_encoder, text_encoder, num_logits=CFG.batch_size,
                        logit_scale=2.30, logit_bias=-10.0)

    siglip_model.compile(optimizer="adam", loss=SigLIPLoss())