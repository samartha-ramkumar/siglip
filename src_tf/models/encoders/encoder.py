import keras_cv
import keras
import keras_nlp
from utils.config import CFG

class ProjectionHead(keras.Model):
    def __init__(
        self,
        embedding_dim=CFG.embedding_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = keras.layers.Dense(embedding_dim)
        self.gelu = keras.layers.Activation("gelu")
        self.fc = keras.layers.Dense(embedding_dim)
        self.dropout = keras.layers.Dropout(dropout)
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
def build_image_encoder():
    backbone = keras_cv.models.ImageClassifier.from_preset(
        CFG.image_preset,
    )
    out = backbone.layers[-2].output
    out = ProjectionHead()(out)
    model = keras.models.Model(backbone.input, out)
    return model

def build_text_encoder():
    backbone = keras_nlp.models.DistilBertClassifier.from_preset(
        CFG.text_preset,
        num_classes=1
    )
    out = backbone.layers[-3].output
    out = ProjectionHead()(out)
    model = keras.models.Model(backbone.input, out)
    return model