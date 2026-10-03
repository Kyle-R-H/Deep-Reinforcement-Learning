import pygame
import Env2048

WIDTH = 500
HEIGHT = 600

# Colours
BACKGROUND = (187, 173, 160)
EMPTY_TILE = (205, 193, 180)
TEXT = (119, 110, 101)
TILE_COLOURS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 194, 46),
    8192: (237, 194, 46),
    16384: (237, 194, 46)
}

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")

clock = pygame.time.Clock()
game = Env2048.Env2048(4, 4)
font = pygame.font.SysFont(
    "arial",
    36,
    bold=True
)

smallFont = pygame.font.SysFont(
    "arial",
    24,
    bold=True
)

# -------------------------
# DRAW BOARD
# -------------------------
def drawBoard():
    screen.fill(BACKGROUND)
    boardSize = 460
    gap = 10
    tileSize = (
        boardSize - gap * (game.cols + 1)
    ) // game.cols

    startX = 20
    startY = 100

    for row in range(game.rows):
        for col in range(game.cols):
            value = game.board[row][col]

            x = startX + col * (tileSize + gap)
            y = startY + row * (tileSize + gap)

            colour = TILE_COLOURS.get(
                value,
                (60, 58, 50)
            )

            pygame.draw.rect(
                screen,
                colour,
                (x, y, tileSize, tileSize),
                border_radius=6
            )

            if value != 0:
                text = font.render(
                    str(value),
                    True,
                    TEXT
                )

                textRect = text.get_rect(
                    center=(
                        x + tileSize // 2,
                        y + tileSize // 2
                    )
                )

                screen.blit(
                    text,
                    textRect
                )

    # TITLE
    title = font.render(
        "2048",
        True,
        (255, 255, 255)
    )

    screen.blit(
        title,
        (20, 30)
    )

    # SCORE
    scoreText = smallFont.render(
        f"Score: {game.score}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        scoreText,
        (150, 40)
    )

    # HIGHEST TILE
    highest = smallFont.render(
        f"Highest: {game.highestVal}",
        True,
        (255, 255, 255)
    )

    screen.blit(
        highest,
        (350, 40)
    )


# -------------------------
# GAME OVER SCREEN
# -------------------------

def drawGameOver():
    drawBoard()
    # Dark transparent overlay
    overlay = pygame.Surface(
        (WIDTH, HEIGHT)
    )

    overlay.set_alpha(180)
    overlay.fill(
        (0, 0, 0)
    )

    screen.blit(
        overlay,
        (0, 0)
    )

    # GAME OVER
    gameOverFont = pygame.font.SysFont(
        "arial",
        50,
        bold=True
    )

    gameOverText = gameOverFont.render(
        "GAME OVER",
        True,
        (255, 255, 255)
    )

    gameOverRect = gameOverText.get_rect(
        center=(
            WIDTH // 2,
            HEIGHT // 2 - 40
        )
    )

    screen.blit(
        gameOverText,
        gameOverRect
    )

    # RESTART
    restartText = smallFont.render(
        "Press R to restart",
        True,
        (255, 255, 255)
    )

    restartRect = restartText.get_rect(
        center=(
            WIDTH // 2,
            HEIGHT // 2 + 20
        )
    )

    screen.blit(
        restartText,
        restartRect
    )

    # QUIT
    quitText = smallFont.render(
        "Press ESC to quit",
        True,
        (255, 255, 255)
    )

    quitRect = quitText.get_rect(
        center=(
            WIDTH // 2,
            HEIGHT // 2 + 55
        )
    )

    screen.blit(
        quitText,
        quitRect
    )


# -------------------------
# MAIN LOOP
# -------------------------
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            # GAME OVER CONTROLS
            if game.isGameOver():
                if event.key == pygame.K_r:
                    game.reset()

                elif event.key == pygame.K_ESCAPE:
                    running = False

                continue

            # NORMAL CONTROLS
            action = None

            if event.key == pygame.K_LEFT:
                action = "a"

            elif event.key == pygame.K_RIGHT:
                action = "d"

            elif event.key == pygame.K_UP:
                action = "w"

            elif event.key == pygame.K_DOWN:
                action = "s"

            elif event.key == pygame.K_r:
                game.reset()

            # MAKE MOVE
            if action is not None:
                moved, reward, validMoves = (
                    game.step(action)
                )
                print(
                    "Action:",
                    action,
                    "| Reward:",
                    reward,
                    "| Valid moves:",
                    validMoves
                )

    # DRAW
    if game.isGameOver():
        drawGameOver()

    else:
        drawBoard()
    pygame.display.flip()
    clock.tick(60)


pygame.quit()