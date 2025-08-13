import importlib


def load_model(c, input_dim, input_channels, output_dim):
  '''
  Import the model module and instantiate
  '''
  print('  | load_model')
  model_module = importlib.import_module('models.' + c.model_type)
  print('  | import module')
  model_class = getattr(model_module, c.model_type)
  print('  | get class method')
  cls = model_class(c, input_dim, input_channels, output_dim)
  print('  | load class')
  return cls


