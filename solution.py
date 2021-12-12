from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from keras.layers import *

class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor # gamma 0.99
        self.model: tf.keras.Model = None
        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # TODO: przygotuj odpowiedni optymizator, pamiętaj o learning rate!
        self.log_action_probability: Optional[tf.Tensor] = None  # zmienna pomocnicza, przyda się do obliczania docelowej straty <=> ln(pi(A|S,w)
        self.tape: Optional[tf.GradientTape] = None  # zmienna pomocnicza, związana z działaniem frameworku
        self.last_error_squared: float = 0.0  # zmienna używana do wizualizacji wyników


    @staticmethod
    def create_actor_critic_model() -> tf.keras.Model:
        # TODO: przygotuj potrzebne warstwy sieci neuronowej o odpowiednich aktywacjach i rozmiarach
        H1 =1024
        H2 = 256
        ACTIONS_SHAPE = 2 # left right
        STATE_SHAPE = 4


        input = Input(shape=(STATE_SHAPE,))  # wejście to pojedynczy stan
        layer_1 = Dense(H1, activation='relu')(input)
        layer_norm = LayerNormalization()(layer_1)
        layer_2 = Dense(H2, activation='relu')(layer_norm)
        layer_norm_2 = LayerNormalization()(layer_2)
        output1 = Dense(ACTIONS_SHAPE, activation='softmax')(layer_norm_2)

        layerA_1 = Dense(H1, activation='relu')(input)
        layerA_norm = LayerNormalization()(layerA_1)
        layerA_2 = Dense(H2, activation='relu')(layerA_norm)
        layerA_norm_2 = LayerNormalization()(layerA_2)
        output2 = Dense(1, activation='linear')(layerA_norm_2)

        return tf.keras.Model(inputs=input, outputs=[output1,output2])

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  # przygotowanie stanu do formatu akceptowanego przez framework

        self.tape = tf.GradientTape()
        with self.tape:
            self.tape.watch(self.model.trainable_weights)
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            # TODO: tu trzeba wybrać odpowiednią akcję korzystając z aktora
            probabilities = self.predict(state)[0]
            distribution = tfp.distributions.Categorical(probs=probabilities)
            action = int(distribution.sample(1)) # Czy aby na pewno sample? Nie lepiej wybrać akcję z największym probablility?
            # ODP NIE, W treści polecenia: pomocnicza klasa, która pozwoli
            # potraktować wartości zwracana przez aktora jak politykę i losować ak1
            # cję do wykonania zgodnie z aktualnym rozkładem ich prawdopodobieństwa;

            self.log_action_probability = tf.math.log(probabilities[action])   # TODO: tu trzeba zapisać do późniejszego wykorzystania logarytm prawdopodobieństwa wybrania uprzednio wybranej akcji (będzie nam potrzebny by policzyć stratę aktora)
        return int(action)

    def predict(self,state):
        output = self.model(state)
        return [tf.reshape(output[0], -1),tf.reshape(output[1], -1)]

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray: # array([[1,2],[3,4]]) -> array([[1, 2, 3, 4]])
        return np.reshape(state, (1, state.size))


def main() -> None:
    environment = gym.make('CartPole-v1')  # zamień na gym.make('LunarLander-v2') by zająć się lądownikiem
    controller = ActorCriticController(environment, 0.00001, 0.99)

    # TUTAJ PRZEŁĄCZ CZY CHCESZ MIEĆ LOSOWE WAGI/CZY ZAŁADOWANE Z ZAPISANEGO NA DYSKU MODELU
    # controller.model = tf.keras.models.load_model("final.model",compile=False)
    controller.model = controller.create_actor_critic_model()

    for i_episode in tqdm(range(2000)):  # tu decydujemy o liczbie epizodów
        done = False
        state = environment.reset()


        while not done:
            environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

            action = controller.choose_action(state)
            new_state, reward, done, info = environment.step(action)
            state = new_state


    environment.close()


if __name__ == '__main__':
    main()
