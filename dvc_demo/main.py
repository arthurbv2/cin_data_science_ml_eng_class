if __name__ == "__main__":

    from nodes.data_acquisition_node import DataAcquisitionNode
    from nodes.data_formatting_node import DataFormattingNode
    from nodes.data_preparation import DataPreprocessing
    from nodes.train_test_split_node import TrainTestSplitNode
    from nodes.training_node import TrainingNode

    """
    queries = [
        'geladeira', 'iphone', 'iphone xr', 'samsung galaxy', 'smartv', 'blu-ray', 'guarda roupa', 'cama',
        'camera', 'sofá', 'mesa de jantar', 'tenis adidas', 'tenis vans', 'camiseta adidas', 'short masculino',
        'short feminino', 'oculos de sol', 'cadeira gamer', 'notebook asus', 'macbook', 'mac', 'alcool em gel',
        'kit de mascara', 'alcool 70', 'notebook', 'pc gamer', 'gin tanqueray', 'vodka', 'celular', 'celulares',
        'smartphone', 'iphone 7', 'raquete de tenis de mesa', 'fogão', 'whisky', 'quadro branco', 'escrivaninha',
        'toalhas de banho', 'pano de chão', 'ferro de passar', 'liquidificador', 'ps4', 'xbox', 'controle playstation',
        'mochila de viagem', 'bola de basquete', 'bola de basquete wilson', 'notebook lenovo', 'máquina de lavar',
        'xiaomi', 'xiomi', 'xaomi', 'ifone', 'controle universal de tv', 'chaveiro', 'headphone', 'headset',
        'fone de ouvido',
        'headset gamer', 'headset bluetooth', 'teclado', 'teclado mecanico', 'smartphone samsung', 'dvd player',
        'vitrola', 'vinho', 'mouse', 'mouse sem fio', 'teclado sem fio', 'guarda chuva', 'chuteira', 'jogo de copos',
        'jogo de taças', 'fogão 4 bocas', 'fogão 6 bocas', 'vaso sanitario', 'box para banheiro'
    ]

    data_node = DataAcquisitionNode(queries)
    data_node.execute()

    dfn = DataFormattingNode(dataset_path=f'database/23_28_13')
    dfn.execute()

    data_preparation = DataPreprocessing('database/refined/21_57_15/data.csv')
    data_preparation.execute()

    ttsn = TrainTestSplitNode(dataset_path='database/pre_processed/22_53_16/data.txt')
    ttsn.execute()
    """

    tn = TrainingNode(dataset_path='database/datasets/01_29_08/', algorithm_id=2, alpha=1.5)
    print(tn.execute())

