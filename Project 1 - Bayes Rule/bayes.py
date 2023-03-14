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

    def sailor_final_location(self, num_search_areas: int) -> int:
        '''Return the actual x,y coordinates of the missing sailor.

        Arguments:
            num_search_areas -- integer number of search areas

        Returns:
            x,y coordinates of the sailor
        '''

        # Find coordinates in respect to any Search Area sub array, as the arrays are the same size.
        self.sailor_actual[0] = np.random.choice(self.sa1.shape[1], 1)
        self.sailor_actual[1] = np.random.choice(self.sa1.shape[0], 1)

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


def main():
    app = Search('Cape Python')
    app.draw_map(last_known=(160, 290))


if __name__ == '__main__':
    main()
