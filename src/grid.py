import torch

class Grid():
    def __init__(self, train_x, num_points=30):
        self.test_grid, self.test_arr = self.grid_maker(train_x, num_points)
    
    # create grid for grid search
    def grid_helper(self, grid_size, num_params, grid_bounds):
        """
        [grid_helper(grid_size, num_params, grid_bounds)] returns a grid of dimensions
        [grid_size] by [num_params], which dictates the parameter space for GP to be 
        conducted over. 
        """
        grid = torch.zeros(grid_size, num_params)
        f_grid_diff = lambda i, x, y : float((x[i][1] - x[i][0]) / (y-2))
        for i in range(num_params):
            grid_diff = f_grid_diff(i, grid_bounds, grid_size)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, 
                                        grid_bounds[i][1] + grid_diff, grid_size)
        return grid

    def grid_maker(self, train_x, num_points=30):
        """
        [grid_maker(train_x, num_points=30)] creates grids to be used for gaussian 
        process predictions. It outputs the dimension of the grid [num_params], 
        paramater space grid [test_grid], and [test_arr].
        """
        # define grid between bounds of RTA time, RTA temp
        num_params = train_x.size(dim=1)
        grid_bounds = [(train_x[:,i].min(), train_x[:,i].max()) for i in range(num_params)]

        # set up test_grid for predictions
        test_grid = self.grid_helper(num_points, num_params, grid_bounds)

        # create n-D grid with n = num_params
        args = (test_grid[:, i] for i in range(num_params))
        test_arr = torch.cartesian_prod(*args)
        return test_grid, test_arr
