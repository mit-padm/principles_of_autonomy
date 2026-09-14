import unittest
import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import functools
import math
import random
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union


from principles_of_autonomy.grader import get_locals

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is present in the autograder
    plt = None


# Parameters of the adversarial domain of 4A. These are handed to the students
# in the notebook and must match the numbers quoted in the 4A prompt.
ADVERSARIAL_BOUNDS = (-2, -3, 5, 4)
ADVERSARIAL_RADIUS = 0.1
ADVERSARIAL_MAX_ITER = 5000
ADVERSARIAL_NUM_TRIALS = 10
ADVERSARIAL_MAX_SUCCESSES = 4
ADVERSARIAL_SEED = 16410


def check_path(path, bounds, environment, start, radius, goal_region):
    """Checks that the path is valid (except for collisions)."""

    minx, miny, maxx, maxy = bounds

    # Check path is a list
    assert isinstance(path, list), "path should be a list."
    assert len(path) > 0, "The returned path shouldn't be empty"

    # Check each element in the path is a tuple and is within bounds
    for i, path_pose in enumerate(path):
        assert isinstance(path_pose, tuple), "Each element of the path should be a tuple, element %d: %s is not." % (
            i, str(path_pose))
        assert len(path_pose) == 2, "Each tuple of the path should have two elements, element %i: %s doesn't" % (
            i, str(path_pose))
        x, y = path_pose
        assert minx <= x <= maxx and miny <= y <= maxy, "Element %i: %s is not within bounds" % (
            i, str(path_pose))

    # Check start of path is start
    startx, starty = path[0]
    assert np.isclose(
        startx, start[0], atol=1e-7), "The start of the path doesn't match the provided start (x is different)"
    assert np.isclose(
        starty, start[1], atol=1e-7), "The start of the path doesn't match the provided start (y is different)"
    # Check end of path is in goal region
    endx, endy = path[-1]
    assert goal_region.contains(
        Point((endx, endy))), "The end of the path should be in the goal region."


def robot_footprint(xy, radius):
    """The collision shape of the robot, as `collision_free` models it."""
    return Point(xy).buffer(radius, resolution=3)


def free_space(bounds, environment, radius):
    """The collision free configuration space of the robot inside `bounds`.

    Growing every obstacle by `radius` (the Minkowski sum of the obstacle and
    the robot) turns collision checking of a disc into a point containment
    test: a robot centre is collision free exactly when it lies outside the
    grown obstacles. `resolution=3` is the same disc approximation that
    `collision_free` and `extend` use, so this agrees with the collision model
    the students implemented.
    """
    free = box(*bounds)
    grown = [obs.buffer(radius, resolution=3) for obs in environment.obstacles]
    if grown:
        free = free.difference(unary_union(grown))
    return free


def free_space_components(free):
    """The connected components of a free space, as a list of Polygons."""
    if free.is_empty:
        return []
    if isinstance(free, Polygon):
        return [free]
    return [geom for geom in free.geoms if isinstance(geom, Polygon)]


def check_path_exists(path_bounds, environment, start, radius, goal_region):
    """Checks a collision free path exists from `start` into `goal_region`.

    This is an exact geometric check on the environment itself, so it does not
    depend on the student's `rrt` being able to find that path.
    """
    minx, miny, maxx, maxy = path_bounds
    startx, starty = start
    assert minx <= startx <= maxx and miny <= starty <= maxy, (
        "The start %s is outside the bounds %s." % (str(start), str(path_bounds)))
    assert goal_region.within(box(*path_bounds).buffer(1e-9)), (
        "The goal region must lie inside the bounds %s." % str(path_bounds))

    footprint = robot_footprint(start, radius)
    for obs in environment.obstacles:
        assert not obs.intersects(footprint), (
            "The robot is already in collision at the start %s." % str(start))

    free = free_space(path_bounds, environment, radius)
    start_point = Point(start)
    # The start may sit exactly on the border of the bounds, so match on
    # distance rather than containment.
    reachable = [comp for comp in free_space_components(free)
                 if comp.distance(start_point) < 1e-9]
    assert reachable, (
        "The start %s is not in the free space of the environment." % str(start))
    for comp in reachable:
        if comp.intersection(goal_region).area > 0.0:
            return
    raise AssertionError(
        "No collision free path exists between the start %s and the goal "
        "region: they are in different connected components of the free "
        "space (or the reachable part of the goal region has no area). The "
        "domain of 4A has to be hard for the RRT, not unsolvable."
        % str(start))


class TestPSet2(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    @timeout_decorator.timeout(5.0)
    def test_01_romania_graph(self):
        romania_graph, romania_locations, romania_connections, Edge = get_locals(
            self.notebook_locals, ["romania_graph", "romania_locations", "romania_connections", "Edge"])
        assert len(romania_graph._nodes) == len(
            romania_locations), "Missing or having extra nodes."
        for node in romania_locations:
            assert node in romania_graph._nodes, "All nodes should be in Romania's graph!"
            assert romania_graph.get_node_pos(
                node) == romania_locations[node], "Location of node is incorrect."
        ne = 0
        assert functools.reduce(
            lambda x, y: x+len(y), romania_graph._edges.values(), 0) == 46, "Num of edges is incorrect."
        for src, edges in romania_connections.items():
            for target, weight in edges.items():
                assert Edge(src, target, weight) in functools.reduce(
                    lambda a, b: a + list(b), romania_graph._edges.values(), []), "Missing or wrong edge."

    @weight(3)
    @timeout_decorator.timeout(5.0)
    def test_02_graph_search_problem(self):
        GraphSearchProblem, romania_graph, SearchNode, romania_connections = get_locals(
            self.notebook_locals, [
                "GraphSearchProblem", "romania_graph", "SearchNode", "romania_connections"]
        )
        test_problem = GraphSearchProblem(romania_graph, "A", "G")
        expanded_sn = test_problem.expand_node(SearchNode("T", "A", 15))
        assert len(expanded_sn) == 2
        assert SearchNode("A") in expanded_sn
        assert SearchNode("L") in expanded_sn
        assert functools.reduce(lambda a, b: a and b._parent == SearchNode(
            "T"), expanded_sn, True), "Wrong parent."
        assert functools.reduce(lambda a, b: a and b._cost == 15 +
                                romania_connections[b.state]["T"], expanded_sn, True), "Wrong accumulated cost."

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_03_uniform_cost_search(self):
        GraphSearchProblem, uniform_cost_search, Path, romania_graph = get_locals(
            self.notebook_locals, ["GraphSearchProblem",
                                   "uniform_cost_search", "Path", "romania_graph"]
        )
        test_problem = GraphSearchProblem(romania_graph, "A", "G")
        result = uniform_cost_search(test_problem)
        assert isinstance(result, tuple) and len(
            result) == 3, "Result should be a 3-tuple"
        assert isinstance(
            result[0], Path), "The first element should be a Path"
        assert result[0].cost == 508, "Cost should be 508."
        assert result[0].path == ['A', 'S', 'R', 'P', 'B', 'G']
        assert result[1] == 15
        assert result[2] == 4

    @weight(3)
    @timeout_decorator.timeout(5.0)
    def test_04_euclid_distance(self):
        eucl_dist = get_locals(self.notebook_locals, ["eucl_dist"])
        assert np.isclose(eucl_dist((-12, 47), (10, -5)), 56.46237685)

    @weight(2)
    @timeout_decorator.timeout(5.0)
    def test_05_euclid_heuristic(self):
        h_to_G, SearchNode = get_locals(
            self.notebook_locals, ["h_to_G", "SearchNode"]
        )
        assert np.isclose(h_to_G(SearchNode("A")), 360.471912914)
        assert np.isclose(h_to_G(SearchNode("Z")), 373.376485601)
        assert np.isclose(h_to_G(SearchNode("H")), 177.991572834)
        assert np.isclose(h_to_G(SearchNode("G")), 0.0)

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_06_greedy(self):
        GraphSearchProblem, romania_graph, h_to_G, greedy_search = get_locals(
            self.notebook_locals, ["GraphSearchProblem",
                                   "romania_graph", "h_to_G", "greedy_search"]
        )
        test_problem = GraphSearchProblem(romania_graph, "A", "G")
        result = greedy_search(test_problem, h_to_G)
        assert result[0].cost == 540, "Cost should be 540."
        assert result[0].path == ['A', 'S', 'F', 'B', 'G']

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_07_astar(self):
        GraphSearchProblem, romania_graph, h_to_G, astar_search = get_locals(
            self.notebook_locals, ["GraphSearchProblem",
                                   "romania_graph", "h_to_G", "astar_search"]
        )
        test_problem = GraphSearchProblem(romania_graph, "A", "G")
        result = astar_search(test_problem, h_to_G)
        assert result[0].cost == 508, "Cost should be 508."
        assert result[0].path == ['A', 'S', 'R', 'P', 'B', 'G']

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_08_grid_to_graph(self):
        Grid, grid_to_graph, Edge = get_locals(
            self.notebook_locals, ["Grid", "grid_to_graph", "Edge"]
        )
        grid_test_str = """0 1 0 1
                        0 0 0 1
                        1 0 0 1
                        0 1 1 0"""
        test_grid = Grid.create_from_str(grid_test_str)
        num_cols, num_rows = test_grid.size
        test_graph = grid_to_graph(test_grid, diagonal_moves=False)

        # Test states
        num_grid_states = np.prod(test_grid.size) - \
            len(test_grid.get_obstacles())
        assert len(test_graph._nodes) == num_grid_states
        for x, y in [(x, y) for x in range(num_cols) for y in range(num_rows)]:
            assert ((x, y) in test_graph) == (
                test_grid.grid_array[x, y] == 0), "({0},{1}) should be in the graph and it's not, or viceversa".format(x, y)
        hori_length, vert_length = test_grid.cell_dimensions
        diag_length = math.sqrt(hori_length**2 + vert_length**2)
        # Check edges - no diagonal
        num_edges = functools.reduce(
            lambda a, b: a+len(b), test_graph._edges.values(), 0)
        assert num_edges == 14
        assert Edge((1, 1), (2, 1),
                    hori_length) in test_graph.node_edges((1, 1))
        assert Edge((1, 1), (1, 2),
                    vert_length) in test_graph.node_edges((1, 1))
        assert Edge((0, 0), (1, 1),
                    diag_length) not in test_graph.node_edges((0, 0))
        # Check edges - diagonal moves
        test_graph = grid_to_graph(test_grid, diagonal_moves=True)
        num_edges = functools.reduce(
            lambda a, b: a+len(b), test_graph._edges.values(), 0)
        assert num_edges == 28
        assert Edge((1, 1), (2, 1),
                    hori_length) in test_graph.node_edges((1, 1))
        assert Edge((1, 1), (1, 2),
                    vert_length) in test_graph.node_edges((1, 1))
        assert Edge((0, 0), (1, 1),
                    diag_length) in test_graph.node_edges((0, 0))

    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_09_eucl_dist_cell(self):
        eucl_dist_cell, SearchNode = get_locals(
            self.notebook_locals, ["eucl_dist_cell", "SearchNode"]
        )
        assert np.isclose(eucl_dist_cell(
            SearchNode((2, 3)), (5, 5), (1, 2)), 5.0)
        assert np.isclose(eucl_dist_cell(
            SearchNode((8, 2)), (4, 2), (2, 3)), 8.0)
        assert np.isclose(eucl_dist_cell(
            SearchNode((3, 4)), (3, 4), (2, 3)), 0.0)

    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_10_astar_grid(self):
        Grid, eucl_dist_cell, grid_to_graph, GraphSearchProblem, astar_search = get_locals(
            self.notebook_locals, [
                "Grid", "eucl_dist_cell", "grid_to_graph", "GraphSearchProblem", "astar_search"]
        )
        start, goal = (2, 1), (4, 3)
        grid_str = """0 0 0 0 0
                    0 1 1 1 0
                    0 1 1 1 1
                    0 1 0 1 0
                    0 0 0 0 0"""
        grid = Grid.create_from_str(grid_str)
        def h(n): return eucl_dist_cell(n, goal, grid.cell_dimensions)
        graph = grid_to_graph(grid, diagonal_moves=True)
        problem = GraphSearchProblem(graph, start, goal)
        solution_astar, expanded, maxq = astar_search(problem, h)
        assert solution_astar.path == [
            (2, 1), (1, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 3)]
        assert np.isclose(solution_astar.cost, 9.65685424949238)

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_11_collision_testing(self):
        collision_free, Environment, Point = get_locals(
            self.notebook_locals, ["collision_free", "Environment", "Point"]
        )
        env = Environment(None)
        # Add some obstacles
        env.add_obstacles([Polygon([(0, 4), (0, 5), (4, 5), (4, 4)]),
                           Polygon([(3, 0), (3, 2), (4, 2), (4, 0)])])

        assert not collision_free((0, 3), 1.5, env)
        assert collision_free((0, 3), 0.5, env)
        assert not collision_free((3.5, 1.5), 0.001, env)
        assert collision_free((0.0, 0.0), 2, env)

    @weight(3)
    @timeout_decorator.timeout(1.0)
    def test_12_nearest_neighbor(self):
        find_nearest_neighbor, Node = get_locals(
            self.notebook_locals, ["find_nearest_neighbor", "Node"]
        )
        list_of_nodes = [
            Node((1, -1)),
            Node((-1, 1)),
            Node((1, 1)),
            Node((-1, -1)),
            Node((1, 0)),  # closest one
        ]

        point = (2, 0)

        output = find_nearest_neighbor(point, list_of_nodes)

        assert (output.xy == (1, 0)).all()

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_13_extend(self):
        extend, Environment, Node = get_locals(
            self.notebook_locals, ["extend", "Environment", "Node"]
        )
        env = Environment(None)
        # Add some obstacles
        env.add_obstacles([Polygon([(0, 4), (0, 5), (4, 5), (4, 4)]),
                           Polygon([(3, 0), (3, 2), (4, 2), (4, 0)])])

        parent = Node((0,0))
        radius = 0.5
        d = 1.0

        xy = (-2, 0)
        new_node = extend(parent, xy, radius, d, env)
        assert isinstance(new_node, Node)
        assert np.allclose(new_node.xy, (-1, 0.0)), "Make sure new nodes are at most D away"
        new_node = extend(parent, (6,0), radius, 8, env)
        assert new_node is None, "Make sure None is returned if the new edge is in collision"
        


    @weight(6)
    @timeout_decorator.timeout(5.0)
    def test_14_simple_environment(self):

        path, bounds, environment, start, radius, goal_region = get_locals(
            self.notebook_locals, ["path_simple", "bounds_simple", "environment_simple",
                                   "start_simple", "radius_simple", "goal_region_simple"]
        )
        check_path(path, bounds, environment, start, radius, goal_region)

    @weight(6)
    @timeout_decorator.timeout(5.0)
    def test_15_bugtrap_environment(self):

        path, bounds, environment, start, radius, goal_region = get_locals(
            self.notebook_locals, ["path_bugtrap", "bounds_bugtrap", "environment_bugtrap",
                                   "start_bugtrap", "radius_bugtrap", "goal_region_bugtrap"]
        )
        check_path(path, bounds, environment, start, radius, goal_region)

    @weight(20)
    @timeout_decorator.timeout(5.0)
    def test_16_complex_environment(self):

        path, bounds, environment, start, radius, goal_region = get_locals(
            self.notebook_locals, ["path_challenging", "bounds_challenging", "environment_challenging",
                                   "start_challenging", "radius_challenging", "goal_region_challenging"]
        )
        check_path(path, bounds, environment, start, radius, goal_region)

    @weight(4)
    @timeout_decorator.timeout(600.0)
    def test_17_adversarial_environment(self):
        """4A: the domain must be solvable, in bounds, and hard for the RRT."""
        rrt, environment, radius, adv_bounds, start, goal_region = get_locals(
            self.notebook_locals, ["rrt", "environment_adversarial",
                                   "radius_adversarial", "bounds_adversarial",
                                   "start_adversarial", "goal_region_adversarial"]
        )
        adv_bounds = tuple(adv_bounds)
        # The given robot radius and bounds must not be changed: widening them
        # would make the domain easier, which is the opposite of the exercise.
        assert isinstance(radius, (int, float)) and np.isclose(
            radius, ADVERSARIAL_RADIUS), (
            "radius_adversarial must stay at %s, got %s." % (
                ADVERSARIAL_RADIUS, str(radius)))
        assert adv_bounds == ADVERSARIAL_BOUNDS, (
            "bounds_adversarial must stay at %s, got %s." % (
                str(ADVERSARIAL_BOUNDS), str(adv_bounds)))
        assert len(environment.obstacles) > 0, (
            "adversarial.yaml doesn't define any obstacles."
        )
        assert isinstance(start, tuple) and len(start) == 2, (
            "start_adversarial should be an (x, y) tuple, got %s." % str(start))
        assert isinstance(goal_region, Polygon) and goal_region.area > 0.0, (
            "goal_region_adversarial should be a shapely Polygon with a "
            "non-zero area, got %s." % str(goal_region))

        # Conditions 1 and 2 of 4A: a collision free path has to exist inside
        # the given bounds. Obstacles are allowed to stick out past the bounds
        # (sealing a passage against the border needs that); the robot never
        # leaves the bounds, so only the free space inside them matters.
        check_path_exists(adv_bounds, environment, start, radius, goal_region)

        # Condition 3 of 4A: over ADVERSARIAL_NUM_TRIALS runs capped at
        # ADVERSARIAL_MAX_ITER samples each, the RRT may succeed at most
        # ADVERSARIAL_MAX_SUCCESSES times.
        successes = 0
        for trial in range(ADVERSARIAL_NUM_TRIALS):
            # Seed both generators so the trials are reproducible whichever
            # source of randomness the student's rrt draws from.
            np.random.seed(ADVERSARIAL_SEED + trial)
            random.seed(ADVERSARIAL_SEED + trial)
            try:
                path = rrt(adv_bounds, environment, start, radius, goal_region,
                           max_iter=ADVERSARIAL_MAX_ITER)
            except TypeError as e:
                raise RuntimeError(
                    "Couldn't run rrt with a max_iter of %d. Your rrt must "
                    "accept a max_iter keyword argument and return None if no "
                    "path was found within that many samples (%s)."
                    % (ADVERSARIAL_MAX_ITER, str(e)))
            finally:
                if plt is not None:
                    # Each rrt call opens a figure; 10 of them would pile up.
                    plt.close("all")
            if path is None:
                continue
            successes += 1
            # A run only counts as a success if it returned a usable path.
            check_path(path, adv_bounds, environment, start, radius, goal_region)

        assert successes <= ADVERSARIAL_MAX_SUCCESSES, (
            "Your RRT solved the adversarial domain %d out of %d times with a "
            "budget of %d samples per run, but 4A allows at most %d "
            "successes. Make the domain harder." % (
                successes, ADVERSARIAL_NUM_TRIALS, ADVERSARIAL_MAX_ITER,
                ADVERSARIAL_MAX_SUCCESSES))

    @weight(5)
    @timeout_decorator.timeout(5.0)
    def test_18_form_word(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("Wall-e".lower())
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
