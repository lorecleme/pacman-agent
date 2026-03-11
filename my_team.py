# LUPASENA AGENT!!!! Lorenzo and Yavuz




# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
import heapq
from contest.game import Actions

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
       
        # You can profile your evaluation time by uncommenting these lines, (lorenzo: does not work, i am overriding the method)
        start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}



class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Beast A* Offensive Agent: basically it has these logics:
    1. GET BOTH FOOD AND CAPSULES, if there are capsules, exclusively target the capsules and possibly eat the enemies
    2. DEFINING THE BORDER, we define the border as the middle of the board
    3. DYNAMIC CARRY LIMIT, default is 3, but if ghosts are scared it goes to 10
    4. BUZZER BEATER TIMEOUT LOGIC, if we have food in our hand and we are close to the border, we run home
    5. EXECUTE A* SEARCH
    """
    def choose_action(self, game_state):
        #print('choosing action')
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1])) 
        my_state = game_state.get_agent_state(self.index)

        # 1. GET BOTH FOOD AND CAPSULES
        food_list = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        
        enemies_to_avoid = []
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        ghosts_are_scared = False
        scared_timers = [enemy.scared_timer for enemy in enemies if not enemy.is_pacman]
        
        if len(scared_timers) > 0 and min(scared_timers) > 5:
            ghosts_are_scared = True

        for enemy in enemies:
            if enemy.is_pacman == False and enemy.scared_timer <= 0:
                enemy_pos = enemy.get_position()
                if enemy_pos is not None:
                    obs_x, obs_y = int(enemy_pos[0]), int(enemy_pos[1])
                    enemies_to_avoid.append((obs_x, obs_y))
                    enemies_to_avoid.append((obs_x + 1, obs_y))
                    enemies_to_avoid.append((obs_x - 1, obs_y))
                    enemies_to_avoid.append((obs_x, obs_y + 1))
                    enemies_to_avoid.append((obs_x, obs_y - 1))

        # 2. DEFINING THE BORDER
        board_width = game_state.get_walls().width
        board_height = game_state.get_walls().height
        if self.index % 2 == 0:  
            border_x = int((board_width / 2) - 1)
        else:  
            border_x = int(board_width / 2)
            
        safe_border_spots = [(border_x, y) for y in range(board_height) if not game_state.has_wall(border_x, y)]

        # 3. DYNAMIC CARRY LIMIT
        carry_limit = 3 # we can change this, depending on how brave the agent is/we feel
        if ghosts_are_scared:
            carry_limit = 10  # Stay and eat out of their food 

        if my_state.num_carrying >= carry_limit or len(food_list) <= 2:
            goal_pos_list = safe_border_spots
        else:
            if len(capsules) > 0:
                # If there are capsules, exclusively target the capsules  the agent is going to be more decisive
                goal_pos_list = capsules
            else:
                # To stop the oscillation, instead of providing ALL food to A* (which makes it indecisive),
                # let's pick just ONE food dot that is furthest away from the visible enemies 
                if len(enemies_to_avoid) > 0:
                    from contest.util import manhattan_distance
                    # Score each food by how far it is from the enemies, minus how far it is from us
                    # We want food that is far from enemies, but relatively close to us
                    best_food = None
                    best_score = -9999
                    for food in food_list:
                        dist_to_enemies = min([manhattan_distance(food, enemy) for enemy in enemies_to_avoid])
                        dist_to_me = self.get_maze_distance(my_pos, food)
                        score = dist_to_enemies - dist_to_me
                        
                        if score > best_score:
                            best_score = score
                            best_food = food
                    
                    goal_pos_list = [best_food] if best_food else food_list
                else:
                    # If we don't see any enemies, just target all food normally
                    goal_pos_list = food_list

            
        # 4. BUZZER BEATER TIMEOUT LOGIC, compute the time left and the minimum distance to the safe spot, so that to make our agent run home if time is almost up
        time_left = game_state.data.timeleft // 4
        min_distance_home = min([self.get_maze_distance(my_pos, safe_spot) for safe_spot in safe_border_spots])

        # Overwrite the goal to explicitly run home if time is almost up
        if my_state.num_carrying > 0 and min_distance_home >= time_left - 3:
            goal_pos_list = safe_border_spots
            
        # 5. EXECUTE A* SEARCH
        if len(goal_pos_list) > 0:
            path = a_star_search(my_pos, game_state, goal_pos_list, enemies_to_avoid)
            if len(path) > 0:
                return path[0]

        from contest.game import Directions
        return Directions.STOP



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A Defensive Agent that patrols its own side and uses A* to intercept invaders.

    Works as follows:
    1. Identify Enemy Invaders
    2. If there are invaders, target them
    3. If there are no invaders, put yourself in between of the border and the big white dot (power capsule) and/or food left.
    """

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_pos = (int(my_pos[0]), int(my_pos[1]))
        
        my_state = game_state.get_agent_state(self.index)

        # 1. Identify Enemy Invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        # We only care about enemies that are currently Pacmans (meaning they are invading our side)
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        
        goal_pos_list = []
        enemies_to_avoid = []

        # 2. Determine our behavior based on invaders
        if len(invaders) > 0:
            # We see invaders? Let's get their exact coordinates
            target_positions = [(int(inv.get_position()[0]), int(inv.get_position()[1])) for inv in invaders]

            if my_state.scared_timer > 0:
                # RUN AWAY
                enemies_to_avoid = target_positions
                
                # basically go back to start position as a safe spot, we might change this later on
                goal_pos_list = [self.start]
            else:
                # if not, let's chase 'em
                goal_pos_list = target_positions
        else:
            # no enemies detected, smart defensive positioning
            
            # first piece of logic: detective
            # has the enemy just eaten something?
            prev_state = self.get_previous_observation()
            if prev_state is not None:
                prev_food = self.get_food_you_are_defending(prev_state).as_list()
                curr_food = self.get_food_you_are_defending(game_state).as_list()
                if len(curr_food) < len(prev_food):
                    # They just ate something, find out what
                    eaten_food = [f for f in prev_food if f not in curr_food]
                    if len(eaten_food) > 0:
                        # go where they ate food
                        goal_pos_list = eaten_food
            
            # second piece of logic: smart patrolling
            # If no food was stolen recently, find the most vulnerable item on our side and put yourself in between of the border and the item
            if len(goal_pos_list) == 0:
                food_to_protect = self.get_food_you_are_defending(game_state).as_list()
                capsules_to_protect = self.get_capsules_you_are_defending(game_state)
                
                # check where the border is
                board_width = game_state.get_walls().width
                board_height = game_state.get_walls().height
                border_x = int((board_width / 2) - 1) if self.index % 2 == 0 else int(board_width / 2)
                
                # if we still have the capsule, go to the middle between the border and the capsule
                if len(capsules_to_protect) > 0:
                    valid_positions = []
                    for x in range(board_width):
                        if (self.index % 2 == 0 and x <= border_x) or (self.index % 2 != 0 and x >= border_x):
                            for y in range(board_height):
                                if not game_state.has_wall(x, y):
                                    valid_positions.append((x, y))

                    goal_pos_list = []
                    for capsule in capsules_to_protect:
                        target_x = (capsule[0] + border_x) // 2
                        target_y = capsule[1]
                        if valid_positions:
                            best_spot = min(valid_positions, key=lambda p: abs(p[0] - target_x) + abs(p[1] - target_y))
                            goal_pos_list.append(best_spot)
                        else:
                            goal_pos_list.append(capsule)
                elif len(food_to_protect) > 0:
                    # find the food that is closest to the centerline (the most vulnerable one)
                    # pick the one with the minimum x-distance from the border
                    vulnerable_food = min(food_to_protect, key=lambda f: abs(f[0] - border_x))
                    goal_pos_list = [vulnerable_food]
                else:
                    goal_pos_list = [self.start] # Fallback

        # use A* to find the path
        path = a_star_search(my_pos, game_state, goal_pos_list, enemies_to_avoid)
        
        # first step
        if len(path) > 0:
            return path[0]

        from contest.game import Directions
        return Directions.STOP


## A star search


class AStarNode():

    def __init__(self, state, parent, action, cost, heuristic):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic
        self.f_cost = cost + heuristic

    def __lt__(self, other):
        return self.f_cost < other.f_cost


# now we need a function that can find the path to the nearest food using A* search


def a_star_search(start_pos, game_state, goal_pos_list, enemies_to_avoid=None):
    """
    Finds the shortest path to any position in goal_pos_list while avoiding enemies_to_avoid.
    Returns a list of actions.
    """
    if enemies_to_avoid is None:
        enemies_to_avoid = []
        
    walls = game_state.get_walls()
    
    # priority Queue for A*
    open_set = []
    
    # Start node: state is an (x,y) tuple. No parent, no action, cost is 0.
    # For now, we will use a heuristic of 0 (which makes this behave like Dijkstra's algorithm).
    start_node = AStarNode(state=start_pos, parent=None, action=None, cost=0, heuristic=0)
    heapq.heappush(open_set, start_node)
    
    # Keep track of positions we've already evaluated
    closed_set = set()

    while open_set:
        # Pop the node with the lowest f_cost
        current_node = heapq.heappop(open_set)
        current_pos = current_node.state

        # did we reach the goal?
        if current_pos in goal_pos_list:
            actions = []
            
            # reconstructing the path

            # looping through the parent nodes until we reach the start node
            node = current_node
            while node.parent is not None:
                actions.append(node.action)
                node = node.parent  # move up the chain to the next parent
            actions.reverse()
            return actions
                
         
            return actions

        # have we already been here?
        if current_pos in closed_set:
            continue
            
        # mark as visited
        closed_set.add(current_pos)

        # generate valid neighbors 
        x, y = current_pos
        
        # Actions is from contest.game. It contains tuples like ('North', (0, 1)) etc.
        # But we can also just manually check the 4 directions.
        possible_moves = [
            ('North', (x, y + 1)),
            ('South', (x, y - 1)),
            ('East',  (x + 1, y)),
            ('West',  (x - 1, y))
        ]
        
        for action, next_pos in possible_moves:
            nx, ny = next_pos
            
           
            # is it a wall? (check walls[nx][ny] - it will be True if it's a wall)
            if walls[nx][ny]:
                continue

            # is it in the enemies_to_avoid list?
            if next_pos in enemies_to_avoid:
                continue
           
            
            # if it is neither a wall nor an enemy:

            new_cost = current_node.cost + 1

            from contest.util import manhattan_distance
            min_dist = min([manhattan_distance(next_pos, goal) for goal in goal_pos_list])
            new_heuristic = min_dist
          
            new_node = AStarNode(state=next_pos, parent=current_node, action=action, cost=new_cost, heuristic=new_heuristic)
            heapq.heappush(open_set, new_node)
     
            pass

    # if we exit the loop and the queue is empty, no path was found
    return []
