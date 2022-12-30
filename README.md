# 2D_random_vs_none_supervise_learning_AI
For study 2D questions, none supervise learning AI and Random ( rims and no rims stages )

#### Consider Food as an objective ####


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

## Result ##

#### Random functions ####

Consider the stage objectives without IF and ELSE you can create input random functions to learn stage rims. 

![Stage explore](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Snake_stage_rims_start_learn_01.mp4.gif?raw=true "Stage explore")

#### None-Supervise AI learning ####

The AI learns to survive with no tight conditions in the stage ( environments ), repeating scenes but different input ( food ) box. ( Inputs are snakes, food and tails removed it tail is fast accelerates )

![Snake AI vs Random](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Snank_AI_vs_Random_1_hour.mp4.gif?raw=true "Snake AI vs Random")

#### Street Fighters ####

Randoms no attacks to prove it can evade more than one round minute.

![Street Fighters sample](https://github.com/jkaewprateep/2D_random_vs_none_supervise_learning_AI/blob/main/Street%20Fighters%20as%20sample.gif?raw=true "Street Fighters sample")
