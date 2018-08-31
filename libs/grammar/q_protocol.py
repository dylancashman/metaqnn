import json

# Define messages to be sent between server and clients
DATESTRING = '%Y-%m-%dT%H:%M:%S.%f'

import datetime
import json
# {"timestamp": "2018-07-23T03:36:36", "values": {"model": {"NNModel": {"_layers": [{"type": "Conv2D", "filters": "64", "kernel_size": "1", "strides": "1"}, {"type": "Conv2D", "filters": "128", "kernel_size": "1", "strides": "1"}, {"type": "Dense", "units": "64"}, {"type": "AveragePooling2D", "pool_size": "2", "strides": "2"}, {"type": "Conv2D", "filters": "64", "kernel_size": "1", "strides": "1"}, {"type": "Dense", "units": "64"}, {"type": "Conv2D", "filters": "128", "kernel_size": "5", "strides": "1"}, {"type": "Conv2D", "filters": "128", "kernel_size": "3", "strides": "1"}, {"type": "Dense", "units": "64"}, {"type": "GlobalAveragePrecision"}], "nparam": "370112"}}}, "uid": "3c25e24e-c0e3-4371-95f8-cc95da3c9ff7", "etime": ["7c63795c-6a52-45f8-a35c-7c1d09e48d11"]}
# {"timestamp": "2018-07-23T04:11:36.196955", "values": {"train_acc": "0.5168"}, "uid": "aba58a84-7daf-46f9-b826-e6021187edad", "etime": ["812f6b96-1a71-4ab5-8a16-af52b1908343", "0", "15"]}


def mQNNLayerToOLayers(layer_string):
    layers = []
    layer = {}
    activation = {}
    layer_args = [ar for ar in layer_string[2:-1].split(',')]
    if (layer_string[0] == 'C'):
        layer['type'] = 'Conv2D'
        layer['filters'] = str(layer_args[0])
        layer['kernel_size'] = str(layer_args[1])
        layer['strides'] = str(layer_args[2])
        layers.append(layer)
#         layers.append(activation)
    elif (layer_string[0] == 'P'):
        layer['type'] = 'AveragePooling2D'
        layer['pool_size'] = str(layer_args[0])
        layer['strides'] = str(layer_args[1])
        layers.append(layer)
#         layers.append(activation)
    elif (layer_string[0] == 'D'):
        layer['type'] = 'Dense'
        layer['units'] = str(4**(int(layer_args[1])))
        layers.append(layer)
#         layers.append(activation)
    elif (layer_string[0:3] == 'GAP'):
        layer['type'] = 'GlobalAveragePrecision'
        layers.append(layer)
#         layers.append(activation)
#     elif (layer_string[0] == 'SM'):
#         
    return layers
        

def mQNNModelToOModel(model_string):
    layers = [layer.strip() for layer in model_string[1:-1].split(', ')]
    flattened = [item for layer in layers for item in mQNNLayerToOLayers(layer)]
    #return [mQNNLayerToOLayers(l) for l in layers]
    return { '_layers': flattened, 'id': net_string_to_id(model_string) }


def parse_message(msg):
    '''takes message with format PROTOCOL and returns a dictionary'''
    return json.loads(msg)

def construct_login_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'login'})

def construct_new_net_message(hostname, net_string, epsilon, iteration_number):
    return json.dumps({'sender': hostname,
                       'type': 'new_net',
                       'net_string': net_string,
                       'epsilon': epsilon,
                       'iteration_number': iteration_number})

def net_string_to_id(net_string):
    return net_string.replace('(', '').replace(')', '').replace(' ', '').replace(',', '-')

def construct_net_trained_message(hostname,
                                  net_string,
                                  acc_best_val,
                                  iter_best_val,
                                  acc_last_val,
                                  iter_last_val,
                                  epsilon,
                                  iteration_number,
                                  logger):
    if (net_string_to_id(net_string) not in logger.models):
        logger.log_model(mQNNModelToOModel(net_string))

    # logger.log_measurements({'id': net_string_to_id(net_string)}, iteration_number, 
    #     {'val_acc': acc_last_val }
    #   )
    logger.save_log()

    return json.dumps({'sender': hostname,
                       'type': 'net_trained',
                       'net_string': net_string,
                       'acc_best_val': acc_best_val,
                       'iter_best_val': iter_best_val,
                       'acc_last_val': acc_last_val,
                       'iter_last_val': iter_last_val,
                       'epsilon': epsilon,
                       'iteration_number': iteration_number})

def construct_net_too_large_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'net_too_large'})

def construct_redundant_connection_message(hostname):
    return json.dumps({'sender': hostname,
                       'type': 'redundant_connection'})
            