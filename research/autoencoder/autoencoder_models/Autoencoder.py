class Encoder(tf.keras.layers.Layer):
    '''Encodes a digit from the MNIST dataset'''
    
    def __init__(self,
                n_dims,
                name='encoder',
                **kwargs):
        super(Encoder,self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.n_layers = 1
        self.encode_layer = layers.Dense(n_dims, activation='relu')
        
    @tf.function        
    def call(self, inputs):
        return self.encode_layer(inputs)
        
        
class Decoder(tf.keras.layers.Layer):
    '''Decodes a digit from the MNIST dataset'''

    def __init__(self,
                n_dims,
                name='decoder',
                **kwargs):
        super(Decoder,self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.n_layers = len(n_dims)
        self.decode_middle = layers.Dense(n_dims[0], activation='relu')
        self.recon_layer = layers.Dense(n_dims[1], activation='sigmoid')
        
    @tf.function        
    def call(self, inputs):
        x = self.decode_middle(inputs)
        return self.recon_layer(x)


class Autoencoder(tf.keras.Model):
    '''Vanilla Autoencoder for MNIST digits'''
    
    def __init__(self,
                 n_dims=[200, 392, 784],
                 name='autoencoder',
                 **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.encoder = Encoder(n_dims[0])
        self.decoder = Decoder([n_dims[1], n_dims[2]])
        
    @tf.function        
    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)
