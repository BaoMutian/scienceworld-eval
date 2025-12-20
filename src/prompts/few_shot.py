"""Few-shot examples for ScienceWorld tasks based on real trajectories."""

# Example 1: Melt task (物态变化 - 融化)
# 真实轨迹：先把水放入冷冻室变成冰，再融化
MELT_EXAMPLE = """
--- Example: Melt Task ---
Goal: Your task is to melt water. First, focus on the substance. Then, take actions that will cause it to change its state of matter.

Observation: This room is called the hallway. You see doors to the green house, living room, art studio, kitchen, bedroom, and workshop.

Think: I need to melt water. Water is usually in the kitchen. Let me go there first.
Action: go to kitchen

Observation: This room is called the kitchen. You see: a thermometer, a counter with a bowl, a freezer (closed), a stove (off), a sink (off), a cupboard (closed).

Think: I need a container for water and to get water. Let me get a pot from the cupboard.
Action: open cupboard

Observation: The cupboard is now open. In the cupboard is: a metal pot, a ceramic cup, a tin cup.

Think: Good, I found a metal pot. Let me pick it up.
Action: pick up metal pot

Observation: You move the metal pot to the inventory.

Think: Now I need to fill it with water from the sink.
Action: move metal pot to sink

Observation: You move the metal pot to the sink.

Think: Activate the sink to fill with water.
Action: activate sink

Observation: The sink is now activated.

Think: Now turn off the sink and get the pot with water.
Action: deactivate sink

Observation: The sink is now deactivated.

Think: Pick up the pot now containing water.
Action: pick up metal pot

Observation: You move the metal pot to the inventory.

Think: To melt something, I first need to freeze it. Let me focus on the water and put it in the freezer.
Action: focus on substance in metal pot

Observation: You focus on the water.

Think: Now put the pot in the freezer to freeze the water into ice.
Action: open freezer

Observation: The freezer is now open.

Think: Move the pot into the freezer.
Action: move metal pot to freezer

Observation: You move the metal pot to the freezer.

Think: Wait for the water to freeze.
Action: wait

Observation: You decide to wait for 10 iterations.

Think: Check if it's frozen using the thermometer.
Action: use thermometer in inventory on substance in metal pot

Observation: the thermometer measures a temperature of -4 degrees celsius

Think: It's frozen (below 0°C). Now I need to melt it by heating. Let me move it to the stove.
Action: pick up metal pot

Observation: You move the metal pot to the inventory.

Think: Move the pot with ice to the stove.
Action: move metal pot to stove

Observation: You move the metal pot to the stove.

Think: Activate the stove to heat and melt the ice.
Action: activate stove

Observation: The stove is now activated.

Think: Wait for the ice to melt.
Action: wait

Observation: You decide to wait for 10 iterations.

Think: Check if it has melted.
Action: use thermometer in inventory on substance in metal pot

Observation: the thermometer measures a temperature of 2 degrees celsius (Task Completed)
"""

# Example 2: Find living thing task (分类任务)
FIND_LIVING_EXAMPLE = """
--- Example: Find Living Thing Task ---
Goal: Your task is to find a(n) living thing. First, focus on the thing. Then, move it to the red box in the kitchen.

Observation: This room is called the hallway. You see: a picture, air, doors to the green house, living room, kitchen, bedroom, and workshop.

Think: I need to find a living thing. Living things include animals and plants. Let me go outside where I might find insects or animals.
Action: go to green house

Observation: You move to the green house.

Think: Let me continue to outside where there may be more living things.
Action: go to outside

Observation: This outside location is called the outside. Here you see: a dove egg, a butterfly egg, a blue jay egg, a fountain with water, an axe, wood, air, a fire pit, the ground.

Think: I see eggs here. A butterfly egg is a living thing. Let me focus on it.
Action: focus on butterfly egg

Observation: You focus on the butterfly egg.

Think: Now pick up the butterfly egg.
Action: pick up butterfly egg

Observation: You move the butterfly egg to the inventory.

Think: I need to move this to the red box in the kitchen. Let me go there.
Action: go to kitchen

Observation: You move to the kitchen.

Think: Now I need to put the butterfly egg in the red box.
Action: move butterfly egg in inventory to red box

Observation: You move the butterfly egg to the red box. (Task Completed)
"""

# Example 3: Use thermometer (测量任务 - 测量未知物质温度)
THERMOMETER_EXAMPLE = """
--- Example: Use Thermometer Task ---
Goal: Your task is to measure the temperature of unknown substance B, which is located around the living room. First, focus on the thermometer. Next, focus on the unknown substance B. If the temperature is above 100.0 degrees, place it in the red box. If below 100.0 degrees, place it in the green box. The boxes are located around the bathroom.

Observation: This room is called the hallway. You see: a picture, air, doors to the green house, living room, kitchen, bedroom, and workshop.

Think: I need a thermometer first. Thermometers are usually in the kitchen.
Action: go to kitchen

Observation: You move to the kitchen. You see: a stove (off), a table with a glass cup, a sink (off), a counter with a bowl, a cupboard (closed), a freezer (closed), a fridge (closed), a thermometer reading 10°C, a lighter, soap, a chair.

Think: I found the thermometer. Let me pick it up.
Action: pick up thermometer

Observation: You move the thermometer to the inventory.

Think: Now focus on the thermometer as the task requires.
Action: focus on thermometer in inventory

Observation: You focus on the thermometer.

Think: Now I need to find unknown substance B in the living room.
Action: go to hallway

Observation: You move to the hallway.

Think: Continue to the living room.
Action: go to living room

Observation: You move to the living room. You see: unknown substance B, air, a chair, a couch with a pillow, a desk with a drawer, a painting.

Think: Found it! Let me pick up the unknown substance B.
Action: pick up unknown substance B

Observation: You move the unknown substance B to the inventory.

Think: Focus on the substance as required.
Action: focus on unknown substance B in inventory

Observation: You focus on the unknown substance B.

Think: Now I need to go to the bathroom where the boxes are to measure and sort.
Action: go to hallway

Observation: You move to the hallway.

Think: Go to kitchen first, then bathroom.
Action: go to kitchen

Observation: You move to the kitchen.

Think: Now go to the bathroom.
Action: go to bathroom

Observation: You move to the bathroom. You see: a toilet, a sink (off), a glass cup, a green box (empty), a bath tub (off), a red box (empty), air.

Think: Now measure the temperature of the unknown substance.
Action: use thermometer in inventory on unknown substance B in inventory

Observation: the thermometer measures a temperature of 29 degrees celsius

Think: 29°C is below 100°C, so I should place it in the green box.
Action: move unknown substance B in inventory to green box

Observation: You move the unknown substance B to the green box. (Task Completed)
"""

# Example 4: Boil task (物态变化 - 沸腾)
BOIL_EXAMPLE = """
--- Example: Boil Task ---
Goal: Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.

Observation: This room is called the hallway. You see: a picture, air, doors to the green house, living room, kitchen, bedroom, and workshop.

Think: I need to boil water. Water and heating equipment are in the kitchen.
Action: go to kitchen

Observation: You move to the kitchen. You see: soap, a painting, a thermometer (10°C), a counter with a bowl, a sink (off), a table with a glass cup, a chair, a freezer (closed), a lighter, a fridge (closed), air, a cupboard (closed), an oven (off, closed), a salt jar, a stove (off).

Think: I'll need a container and water. Let me get a pot from the cupboard.
Action: open cupboard

Observation: The cupboard is now open. In the cupboard is: a tin cup, a ceramic cup, a metal pot, a drawer.

Think: Good, I see a metal pot. Let me pick it up.
Action: pick up metal pot

Observation: You move the metal pot to the inventory.

Think: Now fill the pot with water from the sink.
Action: move metal pot to sink

Observation: You move the metal pot to the sink.

Think: Turn on the sink to fill with water.
Action: activate sink

Observation: The sink is now activated.

Think: Turn off the sink.
Action: deactivate sink

Observation: The sink is now deactivated.

Think: Pick up the pot now containing water.
Action: pick up metal pot

Observation: You move the metal pot to the inventory.

Think: Focus on the water as required by the task.
Action: focus on substance in metal pot

Observation: You focus on the water.

Think: Now put the pot on the stove and heat it.
Action: move metal pot to stove

Observation: You move the metal pot to the stove.

Think: Turn on the stove to start heating.
Action: activate stove

Observation: The stove is now activated.

Think: Wait and monitor the temperature until it boils (100°C).
Action: wait

Observation: You decide to wait for 10 iterations.

Think: Check the temperature.
Action: use thermometer in inventory on substance in metal pot

Observation: the thermometer measures a temperature of 58 degrees celsius

Think: Not boiling yet (need 100°C). Continue waiting.
Action: wait

Observation: You decide to wait for 10 iterations.

Think: Check again.
Action: use thermometer in inventory on substance in metal pot

Observation: the thermometer measures a temperature of 98 degrees celsius (Task Completed)

Think: Almost at boiling point. The task is completed - water has reached boiling temperature and turned to steam.
Action: wait

Observation: The water is now boiling! Steam rises from the pot. (Task Completed)
"""

# Combine all examples for general use
FEW_SHOT_EXAMPLES = f"""
{BOIL_EXAMPLE}

{FIND_LIVING_EXAMPLE}

{THERMOMETER_EXAMPLE}
"""

# Task-specific examples mapping
TASK_TYPE_EXAMPLES = {
    # Phase change tasks (1-1 to 1-4)
    "boil": BOIL_EXAMPLE,
    "melt": MELT_EXAMPLE,
    "freeze": MELT_EXAMPLE,  # Freeze uses similar container/sink logic
    "change-the-state-of-matter-of": BOIL_EXAMPLE,

    # Measurement tasks (2-1 to 2-3)
    "use-thermometer": THERMOMETER_EXAMPLE,
    "measure-melting-point-known-substance": THERMOMETER_EXAMPLE,
    "measure-melting-point-unknown-substance": THERMOMETER_EXAMPLE,

    # Classification tasks (4-1 to 4-4)
    "find-living-thing": FIND_LIVING_EXAMPLE,
    "find-non-living-thing": FIND_LIVING_EXAMPLE,
    "find-plant": FIND_LIVING_EXAMPLE,
    "find-animal": FIND_LIVING_EXAMPLE,
}


def get_task_specific_examples(task_name: str) -> str:
    """Get task-specific few-shot examples.

    Args:
        task_name: Name of the task.

    Returns:
        Task-specific examples or general examples.
    """
    if task_name in TASK_TYPE_EXAMPLES:
        return TASK_TYPE_EXAMPLES[task_name]
    return FEW_SHOT_EXAMPLES
