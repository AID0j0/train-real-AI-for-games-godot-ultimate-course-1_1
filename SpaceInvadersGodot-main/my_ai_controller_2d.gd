extends AIController2D

@onready var REWARDS = {
	&"AG":				"1_PPO",
	&"game_lost":		-1.,
	&"life_lost":		-1.,
	&"game_won":		1.,
	&"alien_shot":		1.,
	# &"action_repeat": is set in the ready function -> this value is bond to the trained model
}
@onready var info_dict_for_python := {"experiment_rewards": REWARDS}
var this_ai_step_rewards := {}

var info_sent_counter := 0

var ingame_points := 0

var move : int
var fire : int
var can_shoot := true

var random_agent := false

var restart_game := false
var restart_game_prev := false
var restart_game_pending := false


@onready var ZERO_OBS = {"obs": "".rpad(84 * 84 * 2, "0")}

func _ready() -> void:
	super._ready()
	
	REWARDS[&"action_repeat"] = get_parent().get_node("Sync").action_repeat
	print("REWARDS in this experiment:")
	print(JSON.stringify(REWARDS, "   ", false))
	if random_agent:
		print("WARNING! USING THE RANDOM AGENT --> the ai will not learn anything")
		

func _process(_delta: float) -> void:
	activate_imgui()
	if restart_game and not restart_game_prev: # this is true for 1 frame only
		restart_game_pending = true
	restart_game_prev = restart_game
	
	restart_game = AiMain.handle_game_reload(restart_game)




func activate_imgui():
	ImGui.Begin("vizualizing inputs")
	if random_agent:
		ImGui.TextWrapped("ATTENTION random agent -> the AI is not learning anyhting!")
	
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
	
	ImGui.TextWrapped("--- this ai step`s reward")
	ImGui.Text(JSON.stringify(this_ai_step_rewards, "   ", false))
	ImGui.End()		
		

func get_obs() -> Dictionary:
	return ZERO_OBS

func give_reward(source: StringName) -> void:
	this_ai_step_rewards[source] = this_ai_step_rewards.get(source, 0.0) + REWARDS[source]

func get_reward() -> float: # gets reset on every exchange between python and godot
	# exachange timing happens after x-frames with x being the action_repeat value
	var total := 0.0 # the decimal is important!
	for value in this_ai_step_rewards.values():
		total += value
	this_ai_step_rewards.clear()	
	return clamp(total, -1, 1)

func get_obs_space() -> Dictionary:
	return {
		"obs": {
			"size": [1, 84, 84], # [channels, height, width]
			"space": "box"
		}
	}

func get_action_space() -> Dictionary:
	return {
		"action": {
			"size": 4,
			"action_type": "discrete"
		}
	}
	
func set_action(action) -> void:	
	if random_agent:
		action = randi_range(0, 3) # picks 0, 1, 2 or 3
	else:
		action = int(action["action"]) # e.g.: { "action": 1.0 } -> 1
	if action == 3:
		move = 1 # 0=LEFT, 1=STAY, 2=RIGHT
		fire = 1 # 0=PEACE, 1=FIRE
	else:
		move = action # 0=LEFT, 1=STAY, 2=RIGHT
		fire = 0 # 0=PEACE, 1=FIRE
			

func get_done() -> bool:
	if restart_game_pending:
		restart_game_pending = false
		return true
	return false

func get_info() -> Dictionary:
	if info_sent_counter >= 11:
		pass
	elif info_sent_counter == 10:
		info_dict_for_python.erase("experiment_rewards")
		info_sent_counter = 11
	else:
		info_sent_counter += 1
	info_dict_for_python["in_game_score"] = ingame_points
	return info_dict_for_python
	
	
	
	
	
	
	
	
	
	
	
	
