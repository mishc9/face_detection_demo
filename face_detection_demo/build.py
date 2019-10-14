import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.applications import MobileNetV2
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

BATCH_SIZE = 64

def weighted_masked_objective(fn):
    """Adds support for masking and sample-weighting to an objective function.
    It transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    # Arguments
        fn: The objective function to wrap,
            with signature `fn(y_true, y_pred)`.
    # Returns
        A function with signature `fn(y_true, y_pred, weights, mask)`.
    """
    if fn is None:
        return None

    def weighted(y_true, y_pred, weights=None, mask=None):
        score_array = fn(y_true, y_pred)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            score_array *= mask
            score_array /= K.mean(mask)

        if weights is not None:
            ndim = K.ndim(score_array)
            weight_ndim = K.ndim(weights)
            score_array = K.mean(score_array,
                                 axis=list(range(weight_ndim, ndim)))
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.mean(score_array)
    return weighted


def loss_angle(class_num, y_true, y_pred, alpha=0.5):
    # cross entropy loss
    idx_tensor = [idx for idx in range(class_num)]
    idx_tensor = tf.Variable(np.array(idx_tensor, dtype=np.float32))
    bin_true = y_true[:, 0]
    cont_true = y_true[:, 1]
    cls_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.cast(bin_true, dtype=tf.int32),
        logits=y_pred)
    pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * idx_tensor, 1) * 3 - 99
    mse_loss = tf.losses.mean_squared_error(labels=cont_true, predictions=pred_cont)
    total_loss = cls_loss + alpha * mse_loss
    return total_loss


def eye_state_clf_head(mob_net_out, dense_units=1024):
    x = layers.GlobalAveragePooling2D()(mob_net_out)
    x = layers.Dropout(0.5, name='eyes_dropout_1')(x)
    x = layers.Dense(dense_units, activation='relu', name='eyes_dense_1')(x)
    x = layers.Dropout(0.5, name='eyes_dropout_2')(x)
    eye_state = layers.Dense(name='eye_state_value', units=1, activation='sigmoid')(x)
    return eye_state


def angles_head(mob_net_out, class_num, dense_units=1024):
    x = layers.GlobalAveragePooling2D()(mob_net_out)
    x = layers.Dropout(0.5, name='angles_dropout_1')(x)
    feature = layers.Dense(units=dense_units, activation='relu')(x)
    feature = layers.Dropout(0.5)(feature)

    fc_yaw = layers.Dense(name='yaw_value', units=class_num)(feature)
    fc_pitch = layers.Dense(name='pitch_value', units=class_num)(feature)
    fc_roll = layers.Dense(name='roll_value', units=class_num)(feature)
    return fc_yaw, fc_pitch, fc_roll


def build_eye_state_and_angle_model(input_shape=(224, 224, 3), batch_size=BATCH_SIZE*2, class_num=66):
    mobilenet = MobileNetV2(include_top=False,
                            input_shape=input_shape,
                            weights='imagenet',
                            )
    mobilenet.trainable = True
    inp = layers.Input(shape=input_shape)
    mob_net_out = mobilenet(inp)
    eye_state = eye_state_clf_head(mob_net_out)
    model = Model(inp, eye_state)

    loss = dict(eye_state_value=binary_crossentropy)

    model.compile(loss=loss,
                  optimizer=Adam(lr=0.0001),
                  metrics=dict(eye_state_value='accuracy'),
                  )
    return model


if __name__ == '__main__':
    model = build_eye_state_and_angle_model()
    model.summary()
