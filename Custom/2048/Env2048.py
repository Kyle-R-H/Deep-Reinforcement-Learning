import random

'''
Everything needed for an RL environment is:
- (current)state
- action
- reward
- next_state
- done(there are no more actions possible)


'''

class Env2048:
    def __init__(self, rows=4, cols=4, seed=None):
        self.rows = rows
        self.cols = cols

        self.rng = random.Random(seed)

        self.actions = ["a", "d", "w", "s"]

        self.validActions = []
        self.board = []
        self.score = 0
        self.reward = 0
        self.highestVal = 0

        self.reset()

    # RESET GAME
    def reset(self):
        self.validActions = []
        
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
        # value = self.rng.choice([2, 2, 2, 4])
        value = self.rng.choice([1, 1, 1, 2])
        self.board[row][col] = value

        if value > self.highestVal:
            self.highestVal = value

    # MERGE ONE LINE
    def mergeLine(self, line):
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

                # newValue = values[i] * 2
                newValue = values[i] + 1
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
    def moveLeft(self):
        for row in range(self.rows):
            self.board[row] = self.mergeLine(
                self.board[row]
            )

    # MOVE RIGHT
    def moveRight(self):
        for row in range(self.rows):
            line = self.board[row][::-1]
            line = self.mergeLine(line)
            self.board[row] = line[::-1]

    # MOVE UP
    def moveUp(self):
        for col in range(self.cols):
            line = [
                self.board[row][col]
                for row in range(self.rows)
            ]

            line = self.mergeLine(line)
            for row in range(self.rows):
                self.board[row][col] = line[row]

    # MOVE DOWN
    def moveDown(self):
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

    # HAS VALID MOVES
    def hasValidMoves(self):
        return len(self.getValidActions()) > 0

    # GAME OVER
    def isGameOver(self):
        return not self.hasValidMoves()

    # SHIFT VALUES
    def step(self, action):
        oldBoard = [
            row[:]
            for row in self.board
        ]

        oldScore = self.score

        if action == "a":
            self.moveLeft()

        elif action == "d":
            self.moveRight()

        elif action == "w":
            self.moveUp()

        elif action == "s":
            self.moveDown()

        else:
            raise KeyError("Invalid action")

        moved = self.board != oldBoard
        reward = self.score - oldScore

        if moved:
            self.addNewValue()

        if self.board != oldBoard:
            self.validActions.append(action)

        # @returns moved: boolean, reward: change in score
        return moved, reward

    # Get Score
    def getScore(self):
        return self.score

    # Get Valid Actions
    def getValidActions(self):
        return self.validActions
    
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