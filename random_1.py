"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/snake.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE
from ple.games.snake import Snake as Snake_Game

from pygame.constants import K_a, K_s, K_d, K_w, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }

nb_frames = 100000000000

global lives
global reward
global steps
global gamescores

action = 0	
steps = 0
lives = 0
reward = 0
gamescores = 0

n_blocks = 64

################ Mixed of data input  ###############
global DATA
DATA = tf.zeros([1, 1, 1, n_blocks * 2 + n_blocks * 3 + 12 ], dtype=tf.float32)
global LABEL
LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)

for i in range(15):
	DATA_row = -9999 * tf.ones([1, 1, 1, n_blocks * 2 + n_blocks * 3 + 12], dtype=tf.float32)		
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
	
for i in range(15):
	DATA_row = 9999 * tf.ones([1, 1, 1, n_blocks * 2 + n_blocks * 3 + 12], dtype=tf.float32)			
	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	
	
DATA = DATA[-30:,:,:,:]
LABEL = LABEL[-30:,:,:,:]
####################################################

momentum = 0.1
learning_rate = 0.0001
batch_size=10

checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not exists(checkpoint_dir) : 
	os.mkdir(checkpoint_dir)
	print("Create directory: " + checkpoint_dir)
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def	read_current_state( string_gamestate ):
	
	GameState = p.getGameState()
	
	if string_gamestate in ['snake_head_x']:
		return GameState[string_gamestate]
	elif string_gamestate in ['snake_head_y']:
		return GameState[string_gamestate]
	elif string_gamestate in ['food_x']:
		return GameState[string_gamestate]
	elif string_gamestate in ['food_y']:
		return GameState[string_gamestate]
	elif string_gamestate in ['snake_body']:
		values = GameState[string_gamestate]
		
		temp = tf.constant( values, shape=( int( len(values) ) ), dtype=tf.float32 )
		
		for i in range( n_blocks - int( len(values) ) ):
			temp = tf.concat([ temp, tf.zeros([1, ], dtype=tf.float32) ], axis=0 )

		temp = tf.constant( temp, shape=( n_blocks ), dtype=tf.float32 )

		return temp.numpy()[0]
		
	elif string_gamestate in ['snake_body_pos']:
		values = GameState[string_gamestate]
		temp = tf.constant( values, shape=( len(values), 2 ), dtype=tf.float32 )
		
		for i in range( n_blocks - int( len(temp) ) ):
			temp = tf.concat([ temp, tf.zeros([1, 2], dtype=tf.float32) ], axis=0 )
		
		temp = tf.constant( temp, shape=( int( len(temp) ) * 2, ), dtype=tf.float32 )
		
		return temp.numpy()[0]
		
	return None

def predict_action( ):
	global DATA
	
	predictions = model.predict(tf.expand_dims(tf.squeeze(DATA), axis=1 ))
	score = tf.nn.softmax(predictions[0])

	return int(tf.math.argmax(score))

def update_DATA( action ):
	global lives
	global reward
	global steps
	global gamescores
	global DATA
	global LABEL
	
	steps = steps + 1
	
	n_steps = abs( steps % ( 6 ) - steps % ( 5 ) )

	list_input = []
	
	info5 = read_current_state('snake_body')
	info6 = read_current_state('snake_body_pos')
	
	info1 = abs( read_current_state('snake_head_x') )
	info2 = abs( 512 - read_current_state('snake_head_y') )
	info3 = abs( read_current_state('food_x') )
	info4 = abs( 512 - read_current_state('food_y') )
	
	distance = ( ( abs( info1 - info3 ) + abs( info2 - info4 ) + abs( info3 - info1 ) + abs( info4 - info2 ) ) / 4 )
	
	contrl = distance + reward
	contr2 = ( info1 - info3 ) + abs( info1 - info3 )
	contr3 = ( info2 - info4 ) + abs( info2 - info4 )
	contr4 = ( info3 - info1 ) + abs( info3 - info1 )
	contr5 = ( info4 - info2 ) + abs( info4 - info2 )
	contr6 = steps
	
	list_input.append( contrl )
	list_input.append( contr2 )
	list_input.append( contr3 )
	list_input.append( contr4 )
	list_input.append( contr5 )
	list_input.append( info1 )
	list_input.append( info2 )
	list_input.append( info3 )
	list_input.append( info4 )
	list_input.append( info5 )
	list_input.append( info6 )
	
	for i in range( ( n_blocks * 2 + n_blocks * 3 + 12 ) - len( list_input ) ):
		list_input.append( 1 )
	
	action_name = list(actions.values())[action]
	action_name = [ x for ( x, y ) in actions.items() if y == action_name]

	print( "steps: " + str( steps ).zfill(6) + " action: " + str(action_name) + " contrl: " + str(int(contrl)).zfill(6) + " contr2: " + str(int(contr2)).zfill(6) + " contr3: " +
			str(int(contr3)).zfill(6) + " contr4: " + str(int(contr4)).zfill(6) + " contr5: " + str(int(contr5)).zfill(6) )
	
	
	print( list_input )
	
	DATA_row = tf.constant([ list_input ], shape=(1, 1, 1, n_blocks * 2 + n_blocks * 3 + 12), dtype=tf.float32)	

	DATA = tf.experimental.numpy.vstack([DATA, DATA_row])
	DATA = DATA[-30:,:,:,:]
	
	LABEL = tf.experimental.numpy.vstack([LABEL, tf.constant(action, shape=(1, 1, 1, 1))])
	LABEL = LABEL[-30:,:,:,:]
	
	DATA = DATA[-30:,:,:,:]
	LABEL = LABEL[-30:,:,:,:]
	
	return DATA, LABEL, steps

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = Snake_Game(width=512, height=512, init_length=3)
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()	# (512, 512, 3)

# {'snake_head_x': 256.0, 'snake_head_y': 256.0, 'food_x': 92, 'food_y': 414, 'snake_body': [0.0, 26.0, 52.0], 'snake_body_pos': [[256.0, 256.0], [230.0, 256.0], [204.0, 256.0]]}

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		# if logs['loss'] <= 0.2 and self.wait > self.patience :
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_shape = (1, n_blocks * 2 + n_blocks * 3 + 12)

model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=input_shape),
	
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5))
model.summary()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Optimizer
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=False,
    name='SGD',
)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Loss Fn
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""								
lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Summary
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer=optimizer, loss=lossfn, metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: FileWriter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if exists(checkpoint_path) :
	model.load_weights(checkpoint_path)
	print("model load: " + checkpoint_path)
	input("Press Any Key!")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
history = model.fit(dataset, epochs=1, callbacks=[custom_callback])
model.save_weights(checkpoint_path)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for i in range(nb_frames):
	
	if p.game_over():
		p.reset_game()
		steps = 0
		lives = 0
		reward = 0
		gamescores = 0
		
	if ( steps == 0 ):
		print('start ... ')
		
	action = predict_action( )
	action_from_list = list(actions.values())[action]

	reward = p.act(action_from_list)
	obs = p.getScreenRGB()
	
	gamescores = gamescores + reward
	
	DATA, LABEL, steps = update_DATA( action )
	
	if ( reward > 0 or steps % 15 == 0  ):
		dataset = tf.data.Dataset.from_tensor_slices((DATA, LABEL))
		history = model.fit(dataset, epochs=2, batch_size=batch_size, callbacks=[custom_callback])
		model.save_weights(checkpoint_path)
