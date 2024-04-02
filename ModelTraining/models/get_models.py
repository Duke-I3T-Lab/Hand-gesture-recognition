def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "PatchTSMixer": PatchTSMixer,
    }
    return models.get(model.lower())(**model_params)