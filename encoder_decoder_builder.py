from keras import Input, Model
from keras.src.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Reshape, UpSampling1D, Conv1DTranspose


def build_encoder(input_row, filters_first_layer, filters_second_layer, kernel_size, dropout):
    x = Conv1D(filters_first_layer, kernel_size, padding='same', activation='relu')(input_row)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters_second_layer, kernel_size, padding='same', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    encoder = Dense(2, activation='softmax')(x)
    return encoder


def build_decoder(input_decoder, encoder, filters_first_layer, filters_second_layer, kernel_size, input_shape):
    d = Dense(64, activation='relu', input_shape=(64,))(input_decoder)
    d = Dense((input_shape[0] // 4) * filters_second_layer, activation='relu')(d)
    d = Reshape((input_shape[0] // 4, filters_second_layer))(d)
    d = UpSampling1D(2)(d)
    d = Conv1DTranspose(filters_second_layer, kernel_size, padding='same', activation='relu')(d)
    d = UpSampling1D(2)(d)
    d = Conv1DTranspose(filters_first_layer, kernel_size, padding='same', activation='relu')(d)
    decoded = Conv1D(1, kernel_size, padding='same', activation='sigmoid')(d)
    return decoded


def build_autoencoder1(filters_first_layer, filters_second_layer, kernel_size, dropout, input_shape):
    input_row = Input(input_shape)

    encoder = build_encoder(input_row, filters_first_layer, filters_second_layer, kernel_size, dropout)
    input_decoder = Input(2)
    decoder = build_decoder(input_decoder, encoder, filters_first_layer, filters_second_layer, kernel_size, input_shape)

    encoder_model = Model(input_row, encoder, name='encoder')
    decoder_model = Model(input_decoder, decoder, name='decoder')
    autoencoder = Model(input_row, decoder_model(encoder_model(input_row)), name="autoencoder")

    encoder_model.summary()
    decoder_model.summary()
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mae')