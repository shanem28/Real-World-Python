import sys
import random
import itertools
import numpy as np
import cv2 as cv

MAP_FILE = 'cape_python.png'

SA1_CORNERS = (130, 265, 180, 315)  # (UL-X, UL-Y, LR-X, LR-Y)
SA2_CORNERS = (80, 255, 130, 305)  # (UL-X, UL-Y, LR-X, LR-Y)
SA3_CORNERS = (105, 205, 155, 255)  # (UL-X, UL-Y, LR-X, LR-Y)


class Search():
    '''Bayesian Search & Rescue game with 3 search areas.'''

    def __init__(self, name: str) -> None:
        self.name = name
        self.img = cv.imread(MAP_FILE, cv.IMREAD_COLOR)
        if self.img is None:
            print(
                f'Could not load map file {MAP_FILE.format(file=sys.stderr)}')
            sys.exit(1)

        self.area_actual = 0
        self.sailor_actual = [0, 0]  # As local coordinates within search area

        # Search Area Sub Array
        self.sa1 = self.img[SA1_CORNERS[1]: SA1_CORNERS[3],
                            SA1_CORNERS[0]: SA1_CORNERS[2]]

        self.sa2 = self.img[SA2_CORNERS[1]: SA2_CORNERS[3],
                            SA2_CORNERS[0]: SA2_CORNERS[2]]

        self.sa3 = self.img[SA3_CORNERS[1]: SA3_CORNERS[3],
                            SA3_CORNERS[0]: SA3_CORNERS[2]]

        # Initial Probabilies
        self.p1 = 0.2
        self.p2 = 0.5
        self.p3 = 0.3

        # Search Effectiveness Probabilities (SEPs)
        self.sep1 = 0
        self.sep2 = 0
        self.sep3 = 0

    def draw_map(self, last_known: tuple):
        '''Display the base map with scale, last known xy location, and search areas.'''

        # Map Scale
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)
        cv.putText(self.img, '0', (8, 370),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.putText(self.img, '50 Nautical Miles', (71, 370),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        # Search area 1
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]),
                     (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(
            self.img, '1', (SA1_CORNERS[0]+3, SA1_CORNERS[1]+15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        # Search area 2
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]),
                     (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(
            self.img, '2', (SA2_CORNERS[0]+3, SA2_CORNERS[1]+15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        # Search area 3
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]),
                     (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(
            self.img, '3', (SA3_CORNERS[0]+3, SA3_CORNERS[1]+15), cv.FONT_HERSHEY_PLAIN, 1, 0)

        # Last known location and legend
        # * OpenCV uses Blue Green Red format instead of RGB

        cv.putText(self.img, '+', (last_known),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = Last Known Position', (274, 355),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # Actual location legend
        cv.putText(self.img, '* = Actual Position', (275, 370),
                   cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        # Draw the window
        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 10)
        cv.waitKey(500)
        # cv.destroyAllWindows()

    def sailor_final_location(self, num_search_areas: int) -> int:
        '''Return the actual x,y coordinates of the missing sailor.

        Arguments:
            num_search_areas -- integer number of search areas

        Returns:
            x,y coordinates of the sailor
        '''

        # Find coordinates in respect to any Search Area sub array, as the arrays are the same size.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1])
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0])

        # Uses a triangle distrabution to determine the area, and 2 will be picked most often (to align with project info)
        area = int(random.triangular(1, num_search_areas + 1))

        # Converts the local coordinates above to global coordinates
        if area == 1:
            x = self.sailor_actual[0] + SA1_CORNERS[0]
            y = self.sailor_actual[1] + SA1_CORNERS[1]
            self.area_actual = 1
        elif area == 2:
            x = self.sailor_actual[0] + SA2_CORNERS[0]
            y = self.sailor_actual[1] + SA2_CORNERS[1]
            self.area_actual = 2
        elif area == 3:
            x = self.sailor_actual[0] + SA3_CORNERS[0]
            y = self.sailor_actual[1] + SA3_CORNERS[1]
            self.area_actual = 3

        return x, y

    def calc_search_effectiveness(self):
        '''Set decimal search effectiveness value per search area'''
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def conduct_search(self, area_num: int, area_array: list, effectiveness_prob: float) -> list:
        '''Return search results and list of coordinates

        Arguments:
            area_num -- integer of current area being searched
            area_array -- array of the current area
            effectiveness_prob -- float related to search effectiveness probability

        Returns:
            Found in Area if found along with the coordinates of the sailor or
            Not Found along with the coordinates. 
        '''
        local_y_range = range(area_array.shape[0])
        local_x_range = range(area_array.shape[1])
        coords = list(itertools.product(local_x_range, local_y_range))
        random.shuffle(coords)
        coords = coords[:int((len(coords) * effectiveness_prob))]
        loc_actual = (self.sailor_actual[0], self.sailor_actual[1])
        if area_num == self.area_actual and loc_actual in coords:
            return f'Found in Area {area_num}', coords
        else:
            return 'Not Found', coords

    def revise_target_probs(self):
        '''
        Update the area target probabilitites based on search effictiveness.
        '''
        denom = self.p1 * (1-self.sep1) + self.p2 * \
            (1-self.sep2) + self.p3 * (1-self.sep3)
        self.p1 = self.p1 * (1-self.sep1) / denom
        self.p2 = self.p2 * (1-self.sep2) / denom
        self.p3 = self.p3 * (1-self.sep3) / denom


def draw_menu(search_num):
    '''Print menu of choices for conducting area searches.'''
    print(f'\nSearch {search_num}')
    print(
        """
        Choose next areas to search:
        0 - Quit
        1 - Search Area 1 twice
        2 - Search Area 2 twice
        3 - Search Area 3 twice
        4 - Search Areas 1 & 2
        5 - Search Areas 1 & 3
        6 - Search Areas 2 & 3
        7 - Start Over
        """
    )


def main():
    app = Search('Cape Python')
    app.draw_map(last_known=(160, 290))
    sailor_x, sailor_y = app.sailor_final_location(num_search_areas=3)
    print('-'*65)
    print('\nInitial Target (P) Probabilities:')
    print(f'P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}')
    search_num = 1

    while True:
        app.calc_search_effectiveness()
        draw_menu(search_num)
        choice = input('Choice: ')
        if choice == "0":
            sys.exit()
        elif choice == "1":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(1, app.sa1, app.sep1)
            app.sep1 = (len(set(coords_1 + coords_2))) / (len(app.sa1)**2)
            app.sep2 = 0
            app.sep3 = 0
        elif choice == "2":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep1 = 0
            app.sep2 = (len(set(coords_1 + coords_2))) / (len(app.sa2)**2)
            app.sep3 = 0
        elif choice == "3":
            results_1, coords_1 = app.conduct_search(3, app.sa3, app.sep3)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
            app.sep2 = 0
            app.sep3 = (len(set(coords_1 + coords_2))) / (len(app.sa3)**2)
        elif choice == "4":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(2, app.sa2, app.sep2)
            app.sep3 = 0
        elif choice == "5":
            results_1, coords_1 = app.conduct_search(1, app.sa1, app.sep1)
            results_2, coords_2 = app.conduct_search(2, app.sa3, app.sep3)
            app.sep2 = 0
        elif choice == "6":
            results_1, coords_1 = app.conduct_search(2, app.sa2, app.sep2)
            results_2, coords_2 = app.conduct_search(3, app.sa3, app.sep3)
            app.sep1 = 0
        elif choice == "7":
            main()
        else:
            print(f"\nSorry, but that isn't a valid choice.", file=sys.stderr)
            continue

        app.revise_target_probs()
        print(f"\nSearch {search_num} Results 1 = {results_1}",
              file=sys.stderr)
        print(f"\nSearch {search_num} Results 2 = {results_2}",
              file=sys.stderr)
        print(f"\nSearch {search_num} Effectiveness (E):")
        print(f"E1 = {app.sep1:.3f}, E2 = {app.sep2:.3f}, E3 = {app.sep3:.3f}")

        if results_1 == 'Not Found' and results_2 == 'Not Found':
            print(
                f"\nNew Target Probabilities (P) for Search {search_num + 1}")
            print(
                f"P1 = {app.p1:.3f}, P2 = {app.p2:.3f}, P3 = {app.p3:.3f}")
        else:
            print(f'Found at x = {sailor_x}, y = {sailor_y}')
            cv.circle(app.img, (sailor_x, sailor_y), 3, (255, 0, 0), 1)
            cv.destroyAllWindows()
            cv.imshow('Search Area', app.img)
            cv.waitKey(500)
            choice = input('Play again (y/n)?: ')
            if choice.upper() == 'Y':
                main()
            else:
                sys.exit()

        search_num += 1


if __name__ == '__main__':
    main()
