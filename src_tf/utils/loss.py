from keras import ops
import keras

class SigLIPLoss(keras.losses.Loss):
    def __init__(self, name="siglip_loss"):
        """Calculates the SigLIP loss.

        Standard sigmoid computes the loss twice, once assuming positive
        labels and once assuming negative ones. But in this case, positives
        are on the "me" diagonal and negatives are elsewhere. So, we only
        compute the loss for each once.

        Call Args:
            y_true: Ground truth labels.
            y_pred: Predicted logits.

        Returns:
            tensor: The SigLIP loss.
        """
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Normalize by the number of positives per column (npos), which is one.
        # Since it's one, we just sum.
        loss = -ops.sum(ops.log_sigmoid(y_true * y_pred), axis=-1)

        # NOTE: This is equivalent to concatenating "me" and "ot" along axis -1 above.
        loss = ops.mean(loss)
        return loss
    
    
class CLIPLoss(keras.losses.Loss):
    def __init__(self, name="clip_loss"):
        """Calculates the CLIP loss.

        Call Args:
            y_true: Ground truth labels.
            y_pred: Predicted logits.

        Returns:
            tensor: The CLIP loss.
        """
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        text_loss = self.cross_entropy(y_true, y_pred)
        image_loss = self.cross_entropy(ops.transpose(y_true), ops.transpose(y_pred))
        loss =  (image_loss + text_loss) / 2.0
        loss = ops.mean(loss)
        return loss
    
    def cross_entropy(self, y_true, y_pred):
        loss = ops.sum(-y_true * ops.log_softmax(y_pred, axis=-1), axis=-1)
        return loss