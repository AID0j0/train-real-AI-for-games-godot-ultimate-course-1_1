extends Node
class_name LifeManager

signal on_life_lost(lifes_left: int)

@export var lifes = 3
@onready var player: Player = $"../Player"
@onready var player_scene = preload("res://Scenes/player.tscn")

@onready var ai_c: Node = AiMain.get_node("AIController2D")

func _ready():
	(player as Player).player_destroyed.connect(on_player_destroyed)
	
func on_player_destroyed():
		lifes -= 1
		ai_c.give_reward(&"life_lost")
		on_life_lost.emit(lifes)
		if lifes != 0:
			player = player_scene.instantiate() as Player
			player.global_position = Vector2(0, 302)
			player.player_destroyed.connect(on_player_destroyed)
			get_tree().root.get_node("main").add_child(player)
		else:
			ai_c.give_reward(&"game_lost")
			

	
