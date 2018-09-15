import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import numpy as np
from PIL import Image
from shutil import copyfile

def save_final_layer_image(image_numpy, image_path):
	# last layer is tanh
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	image_numpy = image_numpy.astype(np.uint8)
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)

def save_channel_image(image_numpy, image_path):
	if image_numpy.shape[0] == 1:
		image_numpy = np.tile(image_numpy, (3, 1, 1))

	# normalize to (0, 1)
	min_value = np.min(image_numpy)
	image_numpy = image_numpy - min_value
	max_value = np.max(image_numpy)
	image_numpy = image_numpy / max_value

	image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
	image_numpy = image_numpy.astype(np.uint8)
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)


def debug_layer_images(target_dir, target_layer):
	"""Save each channel in target_layer as one image into target_dir.

	Args:
		target_dir: str, target directory to save result images.
		target_layer: int, target layer. None for all layers.
	"""

	opt = TestOptions().parse()
	# hard-code some parameters for test
	opt.num_threads = 1   # test code only supports num_threads = 1
	opt.batch_size = 1    # test code only supports batch_size = 1
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True    # no flip
	opt.display_id = -1   # no visdom display
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	model.setup(opt)
	
	# extract each layers in netG_A
	layers = list(list(list(list(model.netG_A.children())[0].children())[0].modules())[0].children())
	
	# print network
	print("--- Start network A->B ---")
	for i, layer in enumerate(layers):
		print("#{}: {}".format(i, layer))
	print("--- End network A->B ---")

	# prepare data: only use first data for test
	data_list_enu = enumerate(dataset)
	i, data = next(data_list_enu)
	print("--- Start data info ---")
	print("data[A].shape: ", data['A'].shape)
	print("A_paths: ", data['A_paths'])
	print("--- End data info ---")
	
	# compute each layers
	output = data['A']
	result = []
	for i in range(len(layers)):
		output = layers[i].cpu()(output)
		print("layer{} output shape: {}".format(i, output.shape))
		result.append(output.detach().numpy())
	
	# create target dir
	if not os.path.exists(target_dir):
		os.makedirs(target_dir)

	# save input image
	path = os.path.join(target_dir, 'input.jpg')
	copyfile(data['A_paths'][0], path)

	# save result image
	path = os.path.join(target_dir, 'output.jpg')
	save_final_layer_image(result[27][0], path)

	# save each layer's each channel as one image

	for i in range(len(layers)):

		if target_layer is not None and target_layer != i:
			continue
		
		print("Create images for layer_{}".format(i))

		layer_path = os.path.join(target_dir, "layer_{}".format(i))
		if not os.path.exists(layer_path):
			os.makedirs(layer_path)
		
		for target_channel in range(result[i].shape[1]):
			path = os.path.join(layer_path, 'channel_{}.jpg'.format(target_channel))
			save_channel_image(result[i][0, target_channel:target_channel + 1], path)
	

if __name__ == '__main__':
	target_dir = 'debug_layer_images'
	target_layer = None
	debug_layer_images(target_dir, target_layer)
