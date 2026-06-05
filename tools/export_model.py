import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO

print("1. Loading PyTorch Model...")
model = PPO.load("radar_guard_policy")
policy = model.policy

print("2. Manually Extracting PyTorch Matrices (Bypassing ONNX)...")
# PyTorch weights are shaped [out, in], but TensorFlow expects [in, out].
# We use .T to transpose them instantly.
w1 = policy.mlp_extractor.policy_net[0].weight.detach().numpy().T
b1 = policy.mlp_extractor.policy_net[0].bias.detach().numpy()

w2 = policy.mlp_extractor.policy_net[2].weight.detach().numpy().T
b2 = policy.mlp_extractor.policy_net[2].bias.detach().numpy()

w_out = policy.action_net.weight.detach().numpy().T
b_out = policy.action_net.bias.detach().numpy()

print("3. Building Pure TensorFlow Clone...")
# Recreate the exact PPO neural network architecture:
# Flatten -> Dense(64, Tanh) -> Dense(64, Tanh) -> Action Logits(2)
tf_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10, 64)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(2)
])

print("4. Injecting Weights...")
# Insert the extracted PyTorch numbers into the empty TensorFlow model
tf_model.layers[1].set_weights([w1, b1])
tf_model.layers[2].set_weights([w2, b2])
tf_model.layers[3].set_weights([w_out, b_out])

print("5. Quantizing directly to TFLite (INT8)...")
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Feed it fake wave data so it learns how to scale the 8-bit integers
def representative_dataset():
    for _ in range(100):
        yield [np.random.uniform(0, 255, size=(1, 10, 64)).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

print("6. Generating C++ Header...")
with open("model_data.cc", "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write("const unsigned char g_model[] = {\n")
    for i, byte in enumerate(tflite_model):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n")

print("Success! 'model_data.cc' is perfectly generated.")