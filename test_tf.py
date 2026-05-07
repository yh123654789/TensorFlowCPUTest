import tensorflow as tf
import time

print("TensorFlow版本:", tf.__version__)

devices = tf.config.list_physical_devices()
print("可用设备:", devices)

a = tf.random.uniform((1000, 1000))
b = tf.random.uniform((1000, 1000))

start = time.time()
c = tf.matmul(a, b)
print(f"矩阵乘法耗时: {time.time() - start:.4f} 秒")

print("TensorFlow CPU 安装成功！")
