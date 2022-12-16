# For Testing

## Dependencies
You have to install packeges below using pip or conda

    pytest
    pytest-timeout  
    pytest-plt

## Run Test

Runs tests seperately depend on markers and save figures at written directories(Recommended)

    pytest -v -m wo_obstacle --plots test_wo_obs_plots --timeout=15
    pytest -v -m with_obstacle --plots test_with_obs_plots --timeout=15
Full test (no result figures, no timeout)

    pytest -v

## Add new test case

to add new test case add new line to @pytest.mark.parametrize

    @pytest.mark.parametrize("plt_name, init_x, init_y, init_theta, init_v, init_w, goal_x, goal_y, heading_cost_weight, obstacle_cost_weight, velocity_cost_weight", [
    ("test_1",0, 0, 0, 0, 0, -0.5, 0.0, 3.1, 1, 1),
    ("test_2",0, 0, 0, 0, 0, 0.5, 0.0, 3.1, 1, 1),
    ("test_3",0, 0, 0, 0, 0, 0.0, -0.5, 3.1, 1, 1),
    ("test_4",0, 0, 0, 0, 0, 0.0, 0.5, 3.1, 1, 1),
    ("test_5",0, 0, 0, 0, 0, -0.5, -0.5, 3.1, 1, 1),
    ("test_6",0, 0, 0, 0, 0, -0.5, 0.5, 3.1, 1, 1),
    ("test_7",0, 0, 0, 0, 0, 0.5, -0.5, 3.1, 1, 1),
    ("test_8",0, 0, 0, 0, 0, 0.5, 0.5, 3.1, 1, 1)
    ])