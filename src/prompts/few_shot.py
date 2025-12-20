"""Few-shot examples for ScienceWorld tasks."""

# Example 1: Melt task (物态变化 - 融化)
MELT_EXAMPLE = """
--- Example: Melt Task ---
Goal: Your task is to melt ice. First, focus on the substance. Then, take actions to cause it to change states of matter.

Observation: You are in the kitchen. In one part of the room is a large kitchen table. On the table is a bowl and glass cup. There is also a stove, fridge, sink, and counter.

Think: I need to melt ice. First, I should find the ice. It's likely in the fridge since that's where ice is typically stored.
Action: open fridge

Observation: The fridge is now open. In the fridge, you see: an ice cube, butter, milk, an orange.

Think: I found the ice cube in the fridge. I should pick it up and focus on it as the task requires.
Action: pick up ice cube

Observation: You pick up the ice cube.

Think: Now I need to focus on the ice cube as required by the task.
Action: focus on ice cube

Observation: You focus on the ice cube.

Think: To melt ice, I need to apply heat. I should put the ice cube on the stove and activate it.
Action: move ice cube to stove

Observation: You move the ice cube to the stove.

Think: Now I need to turn on the stove to heat the ice.
Action: activate stove

Observation: The stove is now activated.

Think: I need to wait for the ice to melt from the heat.
Action: wait

Observation: You wait. The ice cube is melting...

Think: Continue waiting for the melting process to complete.
Action: wait

Observation: The ice cube has melted into water! Task completed.
"""

# Example 2: Find living thing task (分类任务)
FIND_LIVING_EXAMPLE = """
--- Example: Find Living Thing Task ---
Goal: Your task is to find a living thing. First, focus on the thing. Then, move it to the red box.

Observation: You are outside. You can see: a tree, a red box, a rock, a path to the house.

Think: I need to find a living thing. A tree is a living thing (it's a plant). Let me focus on it.
Action: focus on tree

Observation: You focus on the tree. The tree is tall with green leaves.

Think: The tree is too large to move. Let me look for smaller living things. I should explore more.
Action: look around

Observation: Looking around, you see: the tree, a bee flying near some flowers, a rock, a red box.

Think: A bee is a living thing (it's an animal). It would be easier to move than the tree.
Action: pick up bee

Observation: You carefully pick up the bee.

Think: Now I have a living thing. I need to move it to the red box to complete the task.
Action: move bee to red box

Observation: You move the bee to the red box. Task completed!
"""

# Example 3: Use thermometer (测量任务)
THERMOMETER_EXAMPLE = """
--- Example: Use Thermometer Task ---
Goal: Your task is to use the thermometer to measure the temperature of the water in the metal pot.

Observation: You are in the kitchen. On the counter you see: a thermometer, a metal pot containing water. There is also a stove, sink, and fridge.

Think: I need to measure the temperature of water using a thermometer. First, I should pick up the thermometer.
Action: pick up thermometer

Observation: You pick up the thermometer.

Think: Now I need to use the thermometer on the water in the metal pot.
Action: use thermometer on water in metal pot

Observation: The thermometer reads 20°C (room temperature). Task completed!
"""

# Example 4: Boil task (物态变化 - 沸腾)
BOIL_EXAMPLE = """
--- Example: Boil Task ---
Goal: Your task is to boil water. First, focus on the substance. Then, take actions to cause it to change states of matter.

Observation: You are in the kitchen. You can see: a cupboard, a stove, a sink, a counter.

Think: I need to boil water. First, I need to find a container for water and then get water from the sink.
Action: open cupboard

Observation: The cupboard is now open. In the cupboard, you see: a metal pot, a glass cup, a bowl.

Think: A metal pot is good for boiling water on a stove. Let me pick it up.
Action: pick up metal pot

Observation: You pick up the metal pot.

Think: Now I need to fill the pot with water using the sink.
Action: move metal pot to sink

Observation: You move the metal pot to the sink.

Think: Let me activate the sink to fill the pot with water.
Action: activate sink

Observation: The sink is now activated. Water flows into the metal pot.

Think: The pot has water now. Let me pick it up and move it to the stove.
Action: pick up metal pot

Observation: You pick up the metal pot (containing water).

Think: Now I need to focus on the water and then move the pot to the stove to heat it.
Action: focus on water

Observation: You focus on the water in the metal pot.

Think: Let me put the pot on the stove.
Action: move metal pot to stove

Observation: You move the metal pot to the stove.

Think: Now I need to activate the stove to heat the water.
Action: activate stove

Observation: The stove is now activated.

Think: I need to wait for the water to heat up and boil.
Action: wait

Observation: You wait. The water is heating up...

Think: Continue waiting for the water to boil.
Action: wait

Observation: The water is now boiling! Task completed.
"""

# Combine all examples
FEW_SHOT_EXAMPLES = f"""
{MELT_EXAMPLE}

{FIND_LIVING_EXAMPLE}

{THERMOMETER_EXAMPLE}
"""

# Task-specific examples mapping
TASK_TYPE_EXAMPLES = {
    # Phase change tasks (1-1 to 1-4)
    "boil": BOIL_EXAMPLE,
    "melt": MELT_EXAMPLE,
    "freeze": MELT_EXAMPLE,  # Similar process
    "change-the-state-of-matter-of": MELT_EXAMPLE,
    
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

