if __name__ == "__main__":

    from nodes.data_acquisition_node import DataAcquisitionNode

    data_node = DataAcquisitionNode(['iphone', 'xiaomi'])
    data_node.execute()