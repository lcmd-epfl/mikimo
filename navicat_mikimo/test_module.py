from .helper import test_process_data_mkm
from .kinetic_solver import test_add_rate, test_calc_dX_dt, test_get_k

test_add_rate()
test_calc_dX_dt()
test_get_k()
test_process_data_mkm()

if __name__ == '__main__':
    test_add_rate()
    test_calc_dX_dt()
    test_get_k()
    test_process_data_mkm()
