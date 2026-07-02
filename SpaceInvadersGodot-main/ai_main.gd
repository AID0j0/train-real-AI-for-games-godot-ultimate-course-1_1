extends Node



func _ready() -> void:
	pass
	


func _input(event: InputEvent) -> void:
	if Input.is_key_pressed(KEY_R):
		print("================ restarting game because of R key press")
		reload_game()

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
	
	
	
	
	
	
	
	
	
