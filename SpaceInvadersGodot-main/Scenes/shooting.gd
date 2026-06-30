extends Node2D

@onready var ai_c: Node = AiMain.get_node("AIController2D")
var control_mode := {0: "onnx", 1: "human", 2: "training"}
var shoot := false

@export var laser_scene:PackedScene

var can_shoot = true


func _process(delta: float) -> void:
	ai_c.can_shoot = can_shoot
	match control_mode[ai_c.control_mode]:
		"training": shoot = bool(ai_c.fire)
		"human": shoot = Input.is_action_just_pressed("shoot")
	
	if shoot && can_shoot:
		can_shoot = false
		var laser = laser_scene.instantiate() as Area2D
		laser.global_position = get_parent().global_position - Vector2(0, 20)
		get_tree().root.get_node("main").add_child(laser)
		laser.tree_exited.connect(on_laser_destroyed)
 
func on_laser_destroyed():
	can_shoot = true
