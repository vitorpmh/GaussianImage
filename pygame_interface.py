import pygame
import sys
import numpy as np

def pygame_interface():
    # Initialize pygame
    pygame.init()

    # Load an image
    image_path = "/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/001_03_fitting_0.99.png"  # Replace with your image file path
    image = pygame.image.load(image_path)

    # Get image dimensions
    image_width, image_height = image.get_size()

    # Get display size
    display_info = pygame.display.Info()
    screen_width, screen_height = display_info.current_w, display_info.current_h

    # Calculate new dimensions while preserving aspect ratio
    aspect_ratio = image.get_width() / image.get_height()
    if image.get_width() > screen_width or image.get_height() > screen_height - 50:  # Account for button space
        if screen_width / aspect_ratio <= screen_height - 50:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)
        else:
            new_height = screen_height - 300
            new_width = int(new_height * aspect_ratio)
    else:
        new_width, new_height = image.get_width(), image.get_height()

    image = pygame.transform.smoothscale(image, (new_width, new_height))

    # Set up the display
    screen = pygame.display.set_mode((new_width, new_height + 50))  # Add space for the button
    pygame.display.set_caption("Select a Square")

    # Colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Button properties
    button_rect = pygame.Rect(10, new_height + 10, 100, 30)

    # Variables for drawing rectangle
    start_pos = None
    end_pos = None
    selecting = False

    # Main loop
    running = True
    selected_indices = None  # To store the selected coordinates
    while running:
        screen.fill(WHITE)
        screen.blit(image, (0, 0))

        # Draw the button
        pygame.draw.rect(screen, BLUE, button_rect)
        font = pygame.font.Font(None, 24)
        text = font.render("Confirm", True, WHITE)
        screen.blit(text, (button_rect.x + 10, button_rect.y + 5))

        # Draw the selection rectangle
        if start_pos and end_pos:
            x1, y1 = start_pos
            x2, y2 = end_pos
            # Calculate the width and height of the rectangle
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Make the rectangle a square
            size = max(width, height)
            rect = pygame.Rect(min(x1, x2), min(y1, y2), size, size)

            pygame.draw.rect(screen, RED, rect, 2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if button_rect.collidepoint(event.pos):
                        if start_pos and end_pos:
                            x1, y1 = rect.topleft
                            x2, y2 = rect.bottomright
                            # Calculate the scale to map the selection to original image
                            scale_x = image_width / new_width
                            scale_y = image_height / new_height
                            x_min, x_max = sorted([int(x1 * scale_x), int(x2 * scale_x)])
                            y_min, y_max = sorted([int(y1 * scale_y), int(y2 * scale_y)])

                            # Update the selected indices (convert them into a list of coordinates)
                            selected_indices = [(x_min, y_min), (x_max, y_max)]
                            print("Selected indices:", np.array(selected_indices))
                            running = False
                    else:
                        start_pos = event.pos
                        selecting = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    end_pos = event.pos
                    selecting = False

            elif event.type == pygame.MOUSEMOTION:
                if selecting:
                    end_pos = event.pos

        pygame.display.flip()

    pygame.quit()

    return selected_indices
