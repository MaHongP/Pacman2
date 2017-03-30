# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # if legalMoves:
        #     legalMoves = legalMoves.remove('Stop')
        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[
            index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"
        # print 'Chosen Direction', legalMoves[chosenIndex],
        # type(legalMoves[chosenIndex]), legalMoves, scores
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Score = NextState.Score + Sum(FoodDistance) - GhostDistance
        foodDist = []
        ghostDist = []
        foodScore = 0
        ghostScore = 0
        average = (lambda x: sum(x) / float(len(x)))
        for food in newFood.asList():
            foodDist.append(util.manhattanDistance(food, newPos))

        # =======================================================
            # Score Funciton 1

        # print '--->>' * 5
        # print foodDist, newPos, newFood.asList()
        # print '---<<' * 5
        if len(foodDist) > 0:
            foodScore = -average(foodDist)

        if(len(newGhostStates) > 0):
            for ghost in newGhostStates:
                ghostDist.append(util.manhattanDistance(
                    ghost.getPosition(), newPos))
            if min(ghostDist) == 0:
                return -99999
            if min(ghostDist) == 1:
                ghostScore = -100
            else:
                ghostScore = min(ghostDist)

        FinalScore = successorGameState.getScore() + foodScore * 1.5 + ghostScore
        # print newFood.asList(), newPos, foodScore, ghostScore
        return FinalScore

        # # =======================================================
        #     # Score Funciton 2

        # # print '--->>' * 5
        # # print foodDist, newPos, newFood.asList()
        # # print '---<<' * 5
        # if len(foodDist) > 0:
        #     foodScore = -min(foodDist)

        # if(len(newGhostStates) > 0):
        #     for ghost in newGhostStates:
        #         ghostDist.append(util.manhattanDistance(
        #             ghost.getPosition(), newPos))
        #     if min(ghostDist) == 0:
        #         return -99999
        #     if min(ghostDist) == 1:
        #         ghostScore = -100
        #     else:
        #         ghostScore = min(ghostDist)

        # FinalScore = successorGameState.getScore() + foodScore + ghostScore
        # # print newFood.asList(), newPos, foodScore, ghostScore
        # return FinalScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # Return action with highest score
        highestScore = float('-inf')
        # random initialize 
        import math
        legalActions = gameState.getLegalActions(0)
        randIndex = int(math.floor(random.random() * (len(legalActions))))
        nextAction = legalActions[randIndex]

        for action in gameState.getLegalActions(0):
            thisScore = self.minimizer(gameState, gameState.generateSuccessor(
                0, action), self.depth, 0)
            if thisScore > highestScore:
                highestScore = thisScore
                nextAction = action
        print highestScore
        return nextAction

    def maxmizer(self, gameState, state, depth, pacmanId):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        highestScore = float('-inf')
        pacmanId = 0
        for action in state.getLegalActions(pacmanId):
            highestScore = max(
                highestScore,
                self.minimizer(
                    gameState,
                    state.generateSuccessor(pacmanId, action),
                    depth,
                    pacmanId))
        return highestScore

    def minimizer(self, gameState, state, depth, ghostId):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        lowestScore = float('inf')
        ghostId += 1
        for action in state.getLegalActions(ghostId):
            if ghostId < (gameState.getNumAgents() - 1):
                lowestScore = min(
                    lowestScore,
                    self.minimizer(
                        gameState,
                        state.generateSuccessor(ghostId, action),
                        depth,
                        ghostId
                    ))
            else:
                lowestScore = min(
                    lowestScore,
                    self.maxmizer(
                        gameState,
                        state.generateSuccessor(ghostId, action),
                        (depth - 1),
                        ghostId
                    ))
        return lowestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
