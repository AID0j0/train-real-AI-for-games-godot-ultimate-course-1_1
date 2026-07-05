extends Node

class_name PointsCounter

signal on_points_increased(points: int)

var points = 0

@onready var invader_spawner = $"../InvaderSpawner" as InvaderSpawner

@onready var ai_c: Node2D = AiMain.get_node("AIController2D")

func _ready():
	invader_spawner.invader_destroyed.connect(increase_points)
	
	ai_c.ingame_points = points

func increase_points(points_to_add: int):
	points += points_to_add
	on_points_increased.emit(points)
	
	ai_c.ingame_points = points
