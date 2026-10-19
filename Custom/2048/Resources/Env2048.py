import math
import random
from gymnasium import spaces

'''
Everything needed for an RL environment is:
- (current)state
- action
- reward
- next_state
- done(there are no more actions possible)


'''

class Env2048:
    # Action IDs
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    def __init__(self, rows=4, cols=4, seed=None):
        self.rows = rows
        self.cols = cols

        self.seed = seed
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.observation_space = spaces.Box(0, 1, shape=((rows*cols),)) # size of the board

        self.validActions = []
        self.board = []
        self.score = 0
        self.reward = 0
        self.highestVal = 0

        self.reset()

    # RESET GAME
    def reset(self, seed = None):
        self.validActions = []
        
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

        self.board = [
            [0 for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        self.score = 0
        self.reward = 0
        self.highestVal = 0

        self.addNewValue()
        self.addNewValue()

    # ADD NEW TILE
    def addNewValue(self):
        emptyCells = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.board[row][col] == 0
        ]

        if not emptyCells:
            return

        row, col = self.rng.choice(emptyCells)
        value = self.rng.choice([2, 2, 2, 4])
        # value = self.rng.choice([1, 1, 1, 2])
        self.board[row][col] = value

        if value > self.highestVal:
            self.highestVal = value

    # MERGE ONE LINE
    def mergeLine(self, line, check = False):
        if check:
            # Remove empty spaces
                values = [
                    value
                    for value in line
                    if value != 0
                ]
        
                merged = []
                i = 0
        
                while i < len(values):
                    # Two adjacent values can merge
                    if (
                        i + 1 < len(values)
                        and values[i] == values[i + 1]
                    ):
        
                        newValue = values[i] * 2
                        # newValue = values[i] + 1
                        merged.append(newValue)
        
                        i += 2
        
                    else:
                        merged.append(values[i])
                        i += 1
        
                # Add zeros back
                while len(merged) < len(line):
                    merged.append(0)
        
                return merged
        else:
            # Remove empty spaces
            values = [
                value
                for value in line
                if value != 0
            ]

            merged = []
            i = 0

            while i < len(values):
                # Two adjacent values can merge
                if (
                    i + 1 < len(values)
                    and values[i] == values[i + 1]
                ):

                    newValue = values[i] * 2
                    # newValue = values[i] + 1
                    merged.append(newValue)

                    # Add merge value to score
                    self.score += newValue

                    # Track highest tile
                    if newValue > self.highestVal:
                        self.highestVal = newValue

                    i += 2

                else:
                    merged.append(values[i])
                    i += 1

            # Add zeros back
            while len(merged) < len(line):
                merged.append(0)

            return merged

    # MOVE LEFT
    def moveLeft(self, check = False):
        # Update validActions
        if check:
            potentialBoard = [
                row[:]
                for row in self.board
            ]
            
            for row in range(self.rows):
                potentialBoard[row] = self.mergeLine(
                    potentialBoard[row], True
                )
            
            currentBoard = [
                row[:]
                for row in self.board
            ]
            
            if potentialBoard != currentBoard :
                self.setValidActions(self.LEFT)
            
        else: 
            for row in range(self.rows):
                self.board[row] = self.mergeLine(
                    self.board[row]
                )

    # MOVE RIGHT
    def moveRight(self, check = False):
        # Update validActions
        if check:
            potentialBoard = [
                row[:]
                for row in self.board
            ]
            
            for row in range(self.rows):
                line = potentialBoard[row][::-1]
                line = self.mergeLine(line, True)
                potentialBoard[row] = line[::-1]
            
            currentBoard = [
                row[:]
                for row in self.board
            ]
            
            if potentialBoard != currentBoard :
                self.setValidActions(self.RIGHT)
            
        else: 
            for row in range(self.rows):
                line = self.board[row][::-1]
                line = self.mergeLine(line)
                self.board[row] = line[::-1]

    # MOVE UP
    def moveUp(self, check = False):
        # Update validActions
        if check:
            potentialBoard = [
                row[:]
                for row in self.board
            ]
            
            for col in range(self.cols):
                line = [
                    potentialBoard[row][col]
                    for row in range(self.rows)
                ]

                line = self.mergeLine(line, True)
                for row in range(self.rows):
                    potentialBoard[row][col] = line[row]
            
            currentBoard = [
                row[:]
                for row in self.board
            ]
            
            if potentialBoard != currentBoard :
                self.setValidActions(self.UP)
            
        else: 
            for col in range(self.cols):
                line = [
                    self.board[row][col]
                    for row in range(self.rows)
                ]

                line = self.mergeLine(line)
                for row in range(self.rows):
                    self.board[row][col] = line[row]

    # MOVE DOWN
    def moveDown(self, check = False):
        # Update validActions
        if check:
            potentialBoard = [
                row[:]
                for row in self.board
            ]
            
            for col in range(self.cols):
                line = [
                    potentialBoard[row][col]
                    for row in range(self.rows)
                ]
                
                # Invert Array
                line = line[::-1]
                line = self.mergeLine(line, True)
                # Reverse Array
                line = line[::-1]

                for row in range(self.rows):
                    potentialBoard[row][col] = line[row]
            
            currentBoard = [
                row[:]
                for row in self.board
            ]
            
            if potentialBoard != currentBoard :
                self.setValidActions(self.DOWN)
            
        else: 
            for col in range(self.cols):
                line = [
                    self.board[row][col]
                    for row in range(self.rows)
                ]
                
                # Invert Array
                line = line[::-1]
                line = self.mergeLine(line)
                # Reverse Array
                line = line[::-1]

                for row in range(self.rows):
                    self.board[row][col] = line[row]

    # UPDATE VALID ACTIONS
    def updateValidActions(self):
        self.validActions.clear()

        self.moveLeft(True)
        self.moveRight(True)
        self.moveUp(True)
        self.moveDown(True)
        
        # return len(self.validActions()) > 0

    # GAME OVER
    def isGameOver(self):
        return len(self.getValidActions()) <= 0

    # SHIFT VALUES
    def step(self, action):
        oldBoard = [row[:] for row in self.board]
        oldScore = self.score

        if action == self.LEFT:
            # print("Left")
            self.moveLeft()
        elif action == self.RIGHT:
            # print("Right")
            self.moveRight()
        elif action == self.UP:
            # print("Up")
            self.moveUp()
        elif action == self.DOWN:
            # print("Down")
            self.moveDown()
        else:
            raise ValueError(f"Invalid action: {action}")

        moveAvailable = self.board == oldBoard
        reward = self.score - oldScore

        if not moveAvailable:
            self.addNewValue()

        self.updateValidActions()

        terminated = len(self.validActions) == 0
        truncated = False
        info = {}

        return (
            oldBoard,
            self.board,
            terminated,
            truncated,
            reward,
            self.validActions.copy(),
            info
        )

    # Get Score
    def getScore(self):
        return self.score

    # Set Valid Actions
    def setValidActions(self, action):
        return self.validActions.append(action)

    # Get Valid Actions
    def getValidActions(self):
        self.updateValidActions()
        return self.validActions
    # Get Valid Actions
    def getValidActions(env):
        env.updateValidActions()
        return list(env.validActions)
    
    # TOTAL BOARD VALUE
    def getTotalValue(self):
        return sum(
            sum(row)
            for row in self.board
        )

    # EMPTY CELLS
    def getEmptyCells(self):
        return sum(
            value == 0
            for row in self.board
            for value in row
        )

    # MAX TILE
    def getMaxTile(self):
        return max(
            max(row)
            for row in self.board
        )

    def getRows(self):
        return self.rows
    def getCols(self):
        return self.cols
    
    def printBoard(self):
        print(*self.board)

# def startGame(rows = 4, cols = 4, seed= None):
#     Env2048(4, 4)
#     while True:
#         print("Make a move: ")
#         action = input()
#         Env2048.step(action)
        

# def main():
#     print("Starting Game: ")
#     startGame(4, 4)
    
# if __name__ == "__main__":
#     main()