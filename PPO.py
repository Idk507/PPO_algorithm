import tensorflow as tf
import numpy as np
from typing import Dict

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy = self.build_policy_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action_prob(self, state):
        return self.policy.predict(np.array([state]))[0]

    def train(self, states, actions, advantages):
        actions = tf.keras.utils.to_categorical(actions, self.action_dim)
        advantages = np.array(advantages)

        with tf.GradientTape() as tape:
            policy_prob = self.policy(states, training=True)
            chosen_prob = tf.reduce_sum(actions * policy_prob, axis=1)
            old_policy_prob = tf.stop_gradient(chosen_prob)

            ratio = tf.exp(tf.math.log(chosen_prob + 1e-10) - tf.math.log(old_policy_prob + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
