import dataset, model

if __name__ == "__main__":

    ruta_raiz = 'C:/Users/Usuario/Desktop/Universidad/TFG/dataset/labeled'
    train_data = dataset.SocNavHeteroDataset(data_path=ruta_raiz, data_list_file='train_set_socnav3.txt')
    test_data = dataset.SocNavHeteroDataset(data_path=ruta_raiz, data_list_file='test_set_socnav3.txt')
    val_data = dataset.SocNavHeteroDataset(data_path=ruta_raiz, data_list_file='val_set_socnav3.txt')

    # Llamar al modelo (train, test, val) 

    # Mostrar resultados 