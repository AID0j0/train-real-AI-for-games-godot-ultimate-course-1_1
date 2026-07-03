extends Node

var frames_count_per_episode := 0 
var restart_triggered_at_frame_c := -1

func _process(_delta: float) -> void:
	frames_count_per_episode += 1 # an episode = 1 game = 3 lives in SpaceInvaders
	
func handle_game_reload(restart_game):
	if restart_game:
		if restart_triggered_at_frame_c < 0:
			restart_triggered_at_frame_c = frames_count_per_episode
		if frames_count_per_episode - restart_triggered_at_frame_c >= 5:
			frames_count_per_episode = 0
			restart_triggered_at_frame_c = -1
			restart_game = false
			reload_game()
	return restart_game


#func _input(event: InputEvent) -> void:
	#if Input.is_key_pressed(KEY_R):
		#print("================ restarting game because of R key press")
		#reload_game()

func reload_game():
	var old_scene = get_tree().current_scene
	if old_scene:
		old_scene.queue_free()
	
	var root_node = get_tree().root
	var keep_list = ["main", "AiMain", "ImGuiRoot"]
	
	for child in root_node.get_children():
		if not child.name in keep_list:
			child.queue_free()	
	
	await get_tree().process_frame
	
	var scene_ressource = load("res://Scenes/main.tscn")
	var new_instance = scene_ressource.instantiate()
	
	new_instance.name = "main"
	get_tree().root.add_child(new_instance)
	get_tree().current_scene = new_instance
	
	
	
	
	
	
	
	
	
