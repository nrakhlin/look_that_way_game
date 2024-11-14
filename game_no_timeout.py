from enum import Enum
import random
import time

class States(Enum):
    INIT = 1
    USR_POINTER = 2
    USR_LOOK = 3
    GAMEOVER = 4
    LEADERBOARD = 5
    SETTINGS = 6

class Directions():
  left  = "left"
  right = "right"
  up    = "up"
  down  = "down"

  dirs = [left, right, up, down]

  def select_random_direction():
    return Directions.dirs[random.randint(0, 3)]


look = ""

point = ""

state = States.INIT
next_state = None


countdown = 3
score_multiplier = 1
level = 1
score = 0
points_per_score = 1


def get_user_input():
  usr_point = ""
  while(usr_point not in Directions.dirs):
      usr_point = input("Please input your direction for pointing (left, right, up, down): \n")
  time.sleep(0.5)
  ai_point = Directions.select_random_direction()
  print(f"ai point direction: {ai_point}")
  return usr_point, ai_point

while(1):
  print(f"Score: {score}")
  print(f"Level: {level}")
  usr_point = ""
  ai_point = ""
  #currently in state x
  if state == States.INIT:
    print("Welcome to Game")
    pass

  elif state == States.USR_POINTER:
    print("User_pointer")
    usr_point, ai_point = get_user_input()
  
  elif state == States.USR_LOOK:
    print("user_looker")
    usr_point, ai_point = get_user_input()

  elif state == States.GAMEOVER:
    print("Sorry, you loose")
    print(f"Final Score: {score}")
    print(f"Final level: {level}")
    break

  time.sleep(1)

  #handle state transition from 
  if state == States.INIT:
    next_state = States.USR_POINTER
  if state == States.USR_POINTER:
    if usr_point == ai_point:
      score += score_multiplier * points_per_score
      next_state = state
      print("Great job, you gottem! Try to catch this individual once more!!")
    else:
      next_state = States.USR_LOOK
      print("Nice try but you missed, Mr.")
      print("Switch sides!")
  
  if state == States.USR_LOOK:

    if usr_point == ai_point:
      next_state = States.GAMEOVER
      print("The AI got you why would you let it do that!!")
    else:
      countdown = countdown - countdown * 0.2
      score_multiplier = score * 1.1
      level += 1
      next_state = States.USR_POINTER
      print("Nice job, you evaded the AI!!")

  time.sleep(1)
  print("\n")
  
  state = next_state