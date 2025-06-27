import tensorflow as tf

print("TF versão:", tf.__version__)  
print("Build com CUDA:", tf.test.is_built_with_cuda())  
print("GPU disponível (legado):", tf.test.is_gpu_available())  
print("GPUs físicas:", tf.config.list_physical_devices('GPU'))