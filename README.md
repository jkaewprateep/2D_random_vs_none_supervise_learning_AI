# 2D_random_vs_none_supervise_learning_AI
For study 2D questions, none supervise learning AI and Random ( rims and no rims stages )

#### Consider Food as an objective ####

We created the randoms function to see if the target objective is possible or if they are different in target we need to conclude into only one equation, the AI deep-learning model is composed of many layers and functions now we are considering only ```two layers of LSTM```.

#### Challenges ####

1. Snakes not passing rims out of the stages.
2. Snakes do not hit backward themselves, it feedbacks as walls.
3. Snakes eat the food to survive with conditions, we sample to keep away from being too hungry -500 points when each turn players are ```-50 points``` and food collecting only 10 points.
4. Snakes do not turn too fast as randoms do, we can add some delays by simply varying functions but we need to see how different the AI learning and rules conditions are.

👧💬 Can AI learn to simulate the modular function ⁉️
🐑💬 They need to learn from another cutting axis or it's solved that is why modular is very fast and we hear someone use this function in their robots AI competitions, the news tells you that it is cheating when anybody trains for a few days. ( Few years later they also one of the robots competition teams before it stop because it is too hard for a Bachelor degree level at that time to turn to IT computers, hackings, and testing application robots as news Robocons on March 1st )

Creating function, it is the first consideration making X and Y input into our known function response not the computer's monitor ```X =  abs( read_current_state('snake_head_x') )``` and ``` Y = abs( 512 - read_current_state('snake_head_y') )```, monitor pixels. Next, distance is considered a delays buffer as in many games such as car racing and other game every cycle time is accelerated fast with velocity turn output rates, delays are buffers value ( see the delays signals sequence example that is a clock counting backward ). Next is our target action mapping to output channels, all outputs are the same channels but we assume they work separately and Max-Min, and SoftMax select at time approaches. We can discuss whether it is about network latency or separate paths working in IT networks or different conditions environments such as robots. 🐜🐜🐜💬 ```MOD MOD MOD ...``` 🧙💬 ```NO IF``` 🧙💬 ```You shall not pass ‼️ ``` 

```
def random_action( ): 
	
    info1 = abs( read_current_state('snake_head_x') )
    info2 = abs( 512 - read_current_state('snake_head_y') )
    info3 = abs( read_current_state('food_x') )
    info4 = abs( 512 - read_current_state('food_y') )
	
    distance = ( ( abs( info1 - info3 ) + abs( info2 - info4 ) + abs( info3 - info1 ) + abs( info4 - info2 ) ) / 4 )
	
    coeff_01 = distance
    coeff_02 = ( info1 - info3 ) + abs( info1 - info3 )
    coeff_03 = ( info2 - info4 ) + abs( info2 - info4 )
    coeff_04 = ( info3 - info1 ) + abs( info3 - info1 )
    coeff_05 = ( info4 - info2 ) + abs( info4 - info2 )
	
    print( "coeff_01: " + str( coeff_01 ) + " coeff_02: " + str( coeff_02 ) + " coeff_03: " + str( coeff_03 ) + " coeff_04: " 
    	+ str( coeff_04 ) + " coeff_05: " + str( coeff_05 ) 
	)
	
    temp = tf.ones([1, 5], dtype=tf.float32)
    temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05 ], shape=(5, 1), 
    		dtype=tf.float32))
    action = tf.math.argmax(temp, axis=0)

    return int(action[0])
```


#### Consider stage rims as an objective ####

Without IF and ELSE or WHERE conditions, we need the snake's player to work on both conditions eating the food as provided to survive and do not hit the wall when it turns. The easy idea we bring is modular that will raise fast at the target position or value, we can substitute it with multiply by cutting line function but it will create an axis from new function to learn when the ```modular function is working on the same linear function```.

```
def random_action( ): 
	
    info1 = abs( read_current_state('snake_head_x') )
    info2 = abs( 512 - read_current_state('snake_head_y') )
    info3 = abs( read_current_state('food_x') )
    info4 = abs( 512 - read_current_state('food_y') )
	
    distance = ( ( abs( info1 - info3 ) + abs( info2 - info4 ) + abs( info3 - info1 ) + abs( info4 - info2 ) ) / 4 )

    coeff_01 = distance
    coeff_02 = ( info1 - info3 ) + abs( info1 - info3 )
    coeff_03 = ( info2 - info4 ) + abs( info2 - info4 )
    coeff_04 = ( info3 - info1 ) + abs( info3 - info1 )
    coeff_05 = ( info4 - info2 ) + abs( info4 - info2 )
	
    coeff_06 = 15 * 512 / ( info2 % 470 )
    coeff_07 = 15 * 512 / ( ( 512 - info2 ) % 470 )
    coeff_08 = 160 * 512 / ( info1 % 448 )
    coeff_09 = 320 * 512 / ( ( 512 - info1 ) % 447 )
	
    print( "coeff_01: " + str( coeff_01 ) + " coeff_02: " + str( coeff_02 ) + " coeff_03: " + str( coeff_03 ) 
      + " coeff_04: " + str( coeff_04 ) + " coeff_05: " + str( coeff_05 ) 
			+ " coeff_06: " + str( coeff_06 ) + " coeff_07: " + str( coeff_07 ) + " coeff_08: " + str( coeff_08 ) 
      + " coeff_09: " + str( coeff_09 ) + " info2: " + str( info2 ) + " info1: " + str( info1 )
	)
	
    temp = tf.ones([1, 9], dtype=tf.float32)
    temp = tf.math.multiply(temp, tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05, coeff_06, 
                            coeff_07, coeff_08, coeff_09 ], shape=(9, 1), dtype=tf.float32))
    action = tf.math.argmax(temp, axis=0)

    return int(action[0])
```

#### AI inputs ####

Simply input to avoids to make player turns by conditions ```1. It hit it tals``` and ```2. It hit the walls``` and you can try accelerate them by adding some Greedy algorithms by ```counter_clock - scores ```.

```
info1 = abs( read_current_state('snake_head_x') )
info2 = abs( 512 - read_current_state('snake_head_y') )
info3 = abs( read_current_state('food_x') )
info4 = abs( 512 - read_current_state('food_y') )
info5 = read_current_state('snake_body')
info6 = read_current_state('snake_body_pos')

contrl = 1
contr2 = 1
contr3 = 1
contr4 = 1
contr5 = 1
contr6 = gamescores * reward

list_input.append( contrl )
list_input.append( contr2 )
list_input.append( contr3 )
list_input.append( contr4 )
list_input.append( contr5 )
list_input.append( contr6 )
list_input.append( info1 )
list_input.append( info2 )
list_input.append( info3 )
list_input.append( info4 )
list_input.append( info5 )
list_input.append( info6 )
```

## AI networks model ##

Simply inputting all into two layers of LSTM, proves that it is hard to solve in the linear functions single term 🐜💬 ```Fantastics```. 🧙💬 We use ```1 - X```.
🧸💬 Today we see the number one University faculty, many small robots that can learn custom by children that are education technology.

```
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }

input_shape = (1, n_blocks * 2 + n_blocks * 3 + 13)

model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
	
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))

])
		
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5))
model.summary()
```

## Files and Directory ##

| File name | Describtion|
--- | --- |
|sample.py| sample codes |
|random_1.py| sample codes for random condition 1 |
|random_2.py| sample codes for random condition 2 |
|Snake_stage_rims_start_learn_01.mp4.gif| result from learning 1st condition |
|Snank_AI_vs_Random_10_minutes.gif| result from learning for 10 minutes |
|Street Fighters as sample.gif| application with other games, Discrete actions 16 output actions posibility |
|README.md| readme file |

## Result ##

#### Random functions ####

Consider the stage objectives without IF and ELSE you can create input random functions to learn stage rims. 

![Stage explore](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Snake_stage_rims_start_learn_01.mp4.gif?raw=true "Stage explore")

#### None-Supervise AI learning - 10 minutes ####

The AI learns to survive with no tight conditions in the stage ( environments ), repeating scenes but different input ( food ) box. ( Inputs are snakes, food and tails removed it tail is fast accelerates )

![Snake AI vs Random](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Snank_AI_vs_Random_10_minutes.gif?raw=true "Snake AI vs Random")

#### Street Fighters ####

Randoms no attacks to prove it can evade more than one round minute.

![Street Fighters sample](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Street%20Fighters%20as%20sample.gif?raw=true "Street Fighters sample")
