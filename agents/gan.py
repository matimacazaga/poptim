import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

class ClipConstraint(tf.keras.constraints.Constraint):

    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self, weights):
        return {'clip_value': self.clip_value}

class GAN:

    def __init__(self, n_inputs, latent_dim, n_outputs, n_epochs=10000, n_batch=128, plot=False, gan_type='WGAN', n_critic=5):

        self.n_inputs = n_inputs 
        self.latent_dim = latent_dim
        self.n_outputs = n_outputs
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.plot = plot
        self.gan_type = gan_type
        self.n_critic = n_critic

    def wasserstein_loss(self,y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def define_critic(self):

        const = ClipConstraint(0.01)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform', kernel_constraint=const, input_dim= self.n_inputs))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        opt = tf.optimizers.RMSprop(0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)
        return model 

    def define_discriminator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=self.n_inputs))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model 

    def define_generator(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(tf.keras.layers.Dense(self.n_outputs, activation='linear'))
        return model 

    def define_gan(self, generator, discriminator):
        discriminator.trainable = False 
        model = tf.keras.Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def define_wgan(self, generator, critic):
        critic.trainable = False 
        model = tf.keras.Sequential()
        model.add(generator)
        model.add(critic)
        opt = tf.optimizers.RMSprop(0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)
        return model

    def generate_real_samples(self, n):
        X1 = np.random.randn(n) - 0.5 
        X2 = X1 * X1 
        X1 = X1.reshape(n,1)
        X2 = X2.reshape(n,1)
        X = np.hstack((X1, X2))
        if self.gan_type == "WGAN":
            y = - np.ones((n,1))
        else:
            y = np.ones((n,1))
        return X, y

    #def generate_real_samples(self,n):
    #    # Implement it to sample from real stock data. Only 
    #    # parameter is n.
    #    X, y = None, None 
    #    return X, y

    def generate_latent_points(self, n):
        x_input = np.random.randn(self.latent_dim * n)
        x_input = x_input.reshape(n, self.latent_dim)
        return x_input 

    def generate_fake_samples(self, generator, n):
        x_input = self.generate_latent_points(n)
        X = generator.predict(x_input)
        if self.gan_type == 'WGAN':
            y = np.ones((n,1))
        else:
            y = np.zeros((n,1))
        return X, y

    def summarize_performance(self, epoch, generator, discriminator, n=100):
        x_real, y_real = self.generate_real_samples(n)
        x_fake, y_fake = self.generate_fake_samples(generator,n)
        if self.gan_type == "WGAN":
            _, ax = plt.subplots()
            ax.scatter(x_real[:,0], x_real[:,1], color='red')
            ax.scatter(x_fake[:,0], x_fake[:,1], color="blue")
            plt.show()
        else:
            _, acc_real, = discriminator.evaluate(x_real, y_real, verbose=0)
            _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
            print(epoch, acc_real, acc_fake)

        if self.plot:
            plt.scatter(x_real[:,0], x_real[:,1], color='red')
            plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
            plt.show()
    
    def plot_history(self, c1_hist, c2_hist, g_hist):
        _, ax = plt.subplots()
        ax.plot(c1_hist, label='crit_real')
        ax.plot(c2_hist, label='crit_fake')
        ax.plot(g_hist, label='gen')
        ax.legend()
        plt.show()

    def train(self, g_model, d_model, gan_model, n_eval = 2000):
        half_batch = int(self.n_batch/2)
        if self.gan_type == "WGAN":
            c1_hist, c2_hist, g_hist = [], [], []
            for i in range(self.n_epochs):
                c1_tmp, c2_tmp = [], []
                for _ in range(self.n_critic):
                    x_real, y_real = self.generate_real_samples(half_batch)
                    c_loss1 = d_model.train_on_batch(x_real, y_real)
                    c1_tmp.append(c_loss1)
                    x_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
                    c_loss2 = d_model.train_on_batch(x_fake, y_fake)
                    c2_tmp.append(c_loss2)
                c1_hist.append(np.mean(c1_tmp))
                c2_hist.append(np.mean(c2_tmp))
                x_gan = self.generate_latent_points(self.n_batch)
                y_gan = -np.ones((self.n_batch,1))
                g_loss = gan_model.train_on_batch(x_gan, y_gan)
                g_hist.append(g_loss)
                if (i+1)%n_eval == 0:
                    print(f'{i+1}, c1={c1_hist[-1]:.3f}, c2={c2_hist[-1]:.3f}, g={g_loss:.3f}')
            if self.plot:
                self.plot_history(c1_hist, c2_hist, g_hist)
                self.summarize_performance(self.n_epochs, g_model, d_model)
        else:
            for i in range(self.n_epochs):
                x_real, y_real = self.generate_real_samples(half_batch)
                x_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
                d_model.train_on_batch(x_real, y_real)
                d_model.train_on_batch(x_fake, y_fake)
                x_gan = self.generate_latent_points(self.n_batch)
                y_gan = np.ones((self.n_batch, 1))
                gan_model.train_on_batch(x_gan, y_gan)
                if (i+1)%n_eval == 0:
                    self.summarize_performance(i, g_model, d_model)

    def train_gan(self):
        if self.gan_type == "WGAN":
            discriminator = self.define_critic()
        else:
            discriminator = self.define_discriminator()
        generator = self.define_generator()
        gan_model = self.define_gan(generator, discriminator)
        self.train(generator, discriminator, gan_model)