import numpy as np
from IMLearn.metrics.loss_functions import mean_square_error


if __name__ == '__main__':
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011,
                       345856.57131275, 563867.1347574, 395102.94362135])
    print(mean_square_error(y_true, y_pred))