extends AIController2D

# ===========================================================================================================# #===========================================================================================================
@onready var REWARDS = {
	&"AG":          "random",
	&"game_lost":		-0.,
	&"life_lost":		-0.,
	&"game_won":		0.,
	&"alien_shot":		0.,
	&"action_repeat": is set in ready # this is bond to the trained model
}

@onready var info_dict_for_python := {"experiment_rewards": REWARDS}

var info_send_counter := 0

var move : int
var fire : int
var can_shoot := true

var random_agent := true

#===========================================================================================================
func _ready() -> void:
	super._ready() # execute the _ready function from the base skript ai_controller_2d.gd
	for key in REWARDS <------------ fehlt noch 
		if key == &"AG":
			continue
		if not REWARDS[key] is float:
			assert(false, "REWARDS[%s] is not a float, got: %s" % [key, REWARDS[key]])
	
	REWARDS[&"action_repeat"] = get_parent().get_node("Sync").action_repeat
	print("REWARDS in this experiment:")
	print(JSON.stringify(REWARDS, "  ", false))
	if random_agent:
		print("WARNING! USING THE RANDOM AGENT -->> the ai will not learn anything")

func _process(_delta: float) -> void:
	activate_imgui()


func activate_imgui():
	ImGui.Begin("vizualizing inputs")
	var action_name = "if empty imgui crashes"  # 0=LEFT, 1=STAY, 2=RIGHT
	if move == 0:
		action_name = "move_left"
	elif move == 2:
		action_name = "move_right"
	else:
		action_name = "dont move"
	ImGui.TextWrapped(action_name)
	
	var shooting_pressed = "shooting not pressed"
	if fire:
		shooting_pressed = "shooting pressed"
	else:
		shooting_pressed = "shooting not pressed"
	ImGui.TextWrapped(shooting_pressed)
	
	var shooting_allowed = "allowed to shot"
	if can_shoot:
		shooting_allowed = "allowed to shot"
	else:
		shooting_allowed = "not allowed to shot"
	ImGui.TextWrapped(shooting_allowed)
	
	
	ImGui.End()		
		

func get_obs() -> Dictionary:
	#assert(false, "the get_obs method is not implemented when extending from ai_controller") 
	return {"obs":[0, 1, 2]}

func get_reward() -> float:	
	#assert(false, "the get_reward method is not implemented when extending from ai_controller") 
	return 0.0
	
func get_action_space() -> Dictionary:
	#assert(false, "the get get_action_space method is not implemented when extending from ai_controller") 
	return {
		"example_actions_continous" : {
			"size": 2,
			"action_type": "continuous"
		},
		"example_actions_discrete" : {
			"size": 2,
			"action_type": "discrete"
		},
		}
	
func set_action(action) -> void:	
	if not random_agent:
		pass
	else:
		var random_action = randi_range(0, 3) # picks 0, 1, 2 or 3
		if random_action == 3:
			move = 1 # 0=LEFT, 1=STAY, 2=RIGHT
			fire = 1 # 0=PEACE, 1=FIRE
		else:
			move = random_action # 0=LEFT, 1=STAY, 2=RIGHT
			fire = 0 # 0=PEACE, 1=FIRE
			
			
# ===========================================================================================================# #===========================================================================================================
func get_info() -> Dictionary:
    if info_send_counter >= 11:
		pass
	elif info_send_counter == 10:
		info_dict_for_python.erase("experiment_rewards")
		info_send_counter = 11
	else:
		info_send_counter += 1
	return info_dict_for_python
			
			
			
			
			
