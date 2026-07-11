extends Node

@onready var game_viewport: SubViewport = $SubViewportContainer/SubViewport
@onready var main_camera: Camera2D = $Camera2D

@onready var ai_c: Node = AiMain.get_node("AIController2D")

func _ready() -> void:
	game_viewport.world_2d = get_viewport().world_2d
	var sub_camera = Camera2D.new()
	game_viewport.add_child(sub_camera)
	
	## optional things if your camera is more complicated:
	## Sync properties to match the main camera's view
	#sub_camera.position = main_camera.position
	#sub_camera.rotation = main_camera.rotation
	#sub_camera.offset = main_camera.offset
	#sub_camera.anchor_mode = main_camera.anchor_mode
	#sub_camera.ignore_rotation = main_camera.ignore_rotation
	#sub_camera.process_callback = main_camera.process_callback
	#
	## If using drag margins, limits, or smoothing, copy those too
	#sub_camera.drag_horizontal_enabled = main_camera.drag_horizontal_enabled
	#sub_camera.drag_vertical_enabled = main_camera.drag_vertical_enabled
	## ... (add similar lines for drag margins, position smoothing, etc., if enabled on main_camera)
	#
	# ----------

	var main_viewport_size: Vector2 = get_viewport().size
	var sub_viewport_size: Vector2 = game_viewport.size
	sub_camera.zoom = main_camera.zoom * (sub_viewport_size / main_viewport_size)
	
	ai_c.game_viewport = game_viewport
	
	
	
