from .helper import test_process_data_mkm
from .kinetic_solver_runner import (
    test_add_rate,
    test_calc_dX_dt,
    test_get_k,
    test_system_KE_DE,
    test_calc_km,
)


def run_all_tests():
    test_add_rate()
    test_calc_dX_dt()
    test_get_k()
    test_process_data_mkm()
    test_system_KE_DE()
    test_calc_km()


if __name__ == "__main__":
    run_all_tests()
