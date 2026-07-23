extends Area2D

@onready var ai_c: Node = AiMain.get_node("AIController2D")

func _on_area_entered(area):
	if area is Laser: 
		ai_c.give_reward(&"shot_missed")
		area.queue_free()
		
