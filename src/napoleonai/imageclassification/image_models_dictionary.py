import tensorflow as tf

CONF = {
    'EfficientNetV2B3': (tf.keras.applications.efficientnet_v2.EfficientNetV2B3, tf.keras.applications.efficientnet_v2.preprocess_input),
    'Resnet50': (tf.keras.applications.resnet50.ResNet50, tf.keras.applications.resnet50.preprocess_input)
}

