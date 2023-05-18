import pygame
import os

HEX_SIZE = 150
WINDOW_SIZE = 1000
FPS = 60
BACKGROUND_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)

# load assets
assets = {
    'Sheep': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'sheep.png')), (HEX_SIZE, HEX_SIZE)),
    'Wheat': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'wheat.png')), (HEX_SIZE, HEX_SIZE)),
    'Ore': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'ore.png')), (HEX_SIZE, HEX_SIZE)),
    'Brick': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'brick.png')), (HEX_SIZE, HEX_SIZE)),
    'Wood': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'wood.png')), (HEX_SIZE, HEX_SIZE)),
    'Desert': pygame.transform.scale(pygame.image.load(os.path.join('assets', 'desert.png')), (HEX_SIZE, HEX_SIZE)),
}


def catan_hex_positions():
    """
    Generate the hexagon positions for a Settlers of Catan board.
    """
    hex_pattern = [3, 4, 5, 4, 3]
    hex_positions = []
    center_y = WINDOW_SIZE // 2
    center_x = WINDOW_SIZE // 2
    start_x = center_x - HEX_SIZE * 1.5  # Adjust as per your requirements to better center
    start_y = center_y - HEX_SIZE * 2  # Adjust as per your requirements to better center

    for row, num_hex in enumerate(hex_pattern):
        for col in range(num_hex):
            x = start_x + (col * HEX_SIZE)
            y = start_y + (HEX_SIZE * 0.75 * row)
            hex_positions.append((x, y))

        if row < len(hex_pattern) // 2:
            start_x -= HEX_SIZE // 2
        else:
            start_x += HEX_SIZE // 2
    return hex_positions


def draw_board(screen):
    """
    Draws the Settlers of Catan board on the screen.
    """
    resources = ['Sheep', 'Wheat', 'Ore', 'Sheep', 'Sheep', 'Desert', 'Ore', 'Ore', 'Brick', 'Wheat', 'Wood', 'Wood', 'Brick', 'Wheat', 'Sheep', 'Brick', 'Wood', 'Wood', 'Wheat']
    resources_order = [0, 1, 2, 11, 12, 13, 3, 10, 17, 18, 14 ,4, 9, 16, 15, 5, 8, 7, 6]
    hex_positions = catan_hex_positions()

    for i in range(19):
        resource = resources[resources_order[i]]
        img = pygame.image.load(f'assets/{resource}.png')
        img = pygame.transform.scale(img, (HEX_SIZE, HEX_SIZE))
        screen.blit(img, hex_positions[i])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill(BACKGROUND_COLOR)
        draw_board(screen)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()