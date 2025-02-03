import jax.numpy as jnp

from points import update_points_map_batch, mark_duplicates_batched, filter_by_proximity_batch, update_points_map_with_relic_nodes, update_points_map_with_relic_nodes_scan


def test_update_points_map_batch():
    '''
    A simple test for initial conditions and first points gained
    '''
    points_map = jnp.zeros((4, 5, 5))
    positions = jnp.array([
        [
            [-1, -1],
            [-1, -1],
        ],
        [
            [0, 1],
            [4, 3],
        ],
        [
            [3, 2],
            [3, 1],
        ],
        [
            [4, 0],
            [2, 4],
        ]
    ])
    points_gained = jnp.array([3, 1, 2, 0])   

    updated_points_map = update_points_map_batch(points_map, positions, points_gained)

    expected_points_map = jnp.array([
        [
            # should ignore [-1, -1] positions
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.5,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.5],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  0.0,  0.0, -1.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0,  0.0]
        ],
    ])

    assert jnp.array_equal(updated_points_map, expected_points_map)

def test_update_points_map_batch2():
    # 2 confirmed positives, 2 confirmed negative
    points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ]
    ]).repeat(6, axis=0)

    positions = jnp.array([
        [
            # 3 at confirmed positive, 2 at confirmed negative, points gained = 2
            # NO CHANGE
            [2, 0], [2, 3], [3, 2], [2, 3], [-1, -1]
        ],
        [
            # 2 at confirmed positive, 1 at confirmed negative, points gained = 3
            # should update 0.5 for last two points
            [2, 0], [2, 3], [3, 2], [4, 0], [4, 1]
        ],
        [
            # 2 at confirmed positive, 1 at confirmed negative, points gained = 2
            # should update -1.0 for last two points
            [2, 0], [2, 3], [3, 2], [4, 0], [4, 1]
        ],
        [
            # 2 at confirmed positive, 1 at confirmed negative, points gained = 4
            # should update 1.0 for last two points
            [2, 0], [2, 3], [3, 2], [4, 0], [4, 1]
        ],
        [
            # 3 at confirmed negative, points gained = 0
            # should update -1.0 for [0, 0] and [4, 0]
            [-1, -1], [-1, -1], [3, 2], [0, 0], [4, 0]
        ],
        [
            # 4 at confirmed negative, points gained = 1
            # should update 1.0 for [4, 0]
            [-1, -1], [-1, -1], [3, 2], [3, 2], [4, 0]
        ],
    ])
    points_gained = jnp.array([2, 3, 2, 4, 0, 1])

    updated_points_map = update_points_map_batch(points_map, positions, points_gained)

    expected_points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.5],
            [  0.0,  0.0,  0.0,  0.0,  0.5],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0, -1.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  1.0],
            [  0.0,  0.0,  0.0,  0.0,  1.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [ -1.0,  0.0,  1.0,  0.0, -1.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  1.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ]
    ])
    assert jnp.allclose(updated_points_map, expected_points_map)

def test_update_points_map_batch3():
    # 3 confirmed positives, 4 confirmed negative
    points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0, -1.0,  0.0,  0.0,  0.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0, -1.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0,  0.0]
        ]
    ]).repeat(4, axis=0)

    positions = jnp.array([
        [
            # 3 at confirmed positive, 1 at confirmed negative, 3 unconfirmed, points gained = 2
            # should update -1.0 for last three points
            [2, 0], [2, 0], [0, 2], [0, 2], [4, 1], [3, 3], [3, 4]
        ],
        [
            # 3 at confirmed positive, 3 unconfirmed, points gained = 4
            # should update 0.3333 for last three points
            [2, 0], [2, 3], [-1, -1], [0, 2], [4, 1], [3, 3], [3, 4]
        ],
        [
            # 3 at confirmed positive, 1 at confirmed negative, 3 unconfirmed, points gained = 5
            # should update 0.6667 for last three points
            [2, 0], [2, 3], [3, 2], [0, 2], [4, 1], [3, 3], [3, 4]
        ],
        [
            # 5 at confirmed positive, 2 at confirmed negative, points gained = 0
            # simulate energy depletion => should remain same point map
            [2, 0], [2, 3], [3, 2], [0, 2], [2, 0], [2, 3], [2, 0]
        ],
    ])
    points_gained = jnp.array([2, 4, 5, 0])

    updated_points_map = update_points_map_batch(points_map, positions, points_gained)

    expected_points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0, -1.0,  0.0,  0.0, -1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0, -1.0,  1.0, -1.0,  0.0],
            [  0.0,  0.0, -1.0, -1.0,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0, -1.0,  0.0,  0.0,  0.3333],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0, -1.0,  1.0,  0.3333,  0.0],
            [  0.0,  0.0, -1.0,  0.3333,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0, -1.0,  0.0,  0.0,  0.6667],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0, -1.0,  1.0,  0.6667,  0.0],
            [  0.0,  0.0, -1.0,  0.6667,  0.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0, -1.0,  0.0,  0.0,  0.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0, -1.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0,  0.0]
        ]
    ])
    assert jnp.allclose(updated_points_map, expected_points_map)

def test_update_points_map_batch4():
    # 3 confirmed positives, 4 confirmed negative
    points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.3333],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.3333,  0.0],
            [  0.0,  0.0,  0.0,  0.3333, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.6667],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.6667,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.6667],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.6667,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.6667],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.6667,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
    ])

    positions = jnp.array([
        [
            # 3 at confirmed positive, 2 at confirmed negative, 2 unconfirmed, points gained = 4
            # should update 0.5 for [4, 1] and [3, 4]
            [2, 0], [2, 3], [3, 2], [0, 2], [4, 1], [3, 2], [3, 4]
        ],
        [
            # 3 at confirmed positive, 2 at confirmed negative, 2 unconfirmed, points gained = 5
            # should update 1.0 for [4, 1] and [3, 4]
            [2, 0], [2, 3], [3, 2], [0, 2], [4, 1], [3, 2], [3, 4]
        ],
        [
            # 3 at confirmed positive, 2 at confirmed negative, 2 unconfirmed, points gained = 3
            # should update -1.0 for [4, 1] and [3, 3]
            [2, 0], [2, 3], [3, 2], [0, 2], [4, 1], [3, 2], [3, 3]
        ],
        [
            # 3 at confirmed positive, 2 at confirmed negative, 2 unconfirmed, points gained = 3
            # should update -1.0 for [4, 1] and [3, 3]
            [2, 0], [2, 3], [3, 2], [0, 2], [4, 1], [3, 2], [3, 3]
        ],
    ])
    points_gained = jnp.array([4, 5, 3, 3])

    updated_points_map = update_points_map_batch(
        points_map,
        mark_duplicates_batched(positions),
        points_gained
    )

    expected_points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.5],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.3333,  0.0],
            [  0.0,  0.0,  0.0,  0.5, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.6667,  0.0],
            [  0.0,  0.0,  0.0,  1.0,  -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
    ])
    assert jnp.allclose(updated_points_map, expected_points_map)

    positions2 = jnp.array([
        [
            # 4 units at 2 confirmed positive, 2 at confirmed negative, points gained = 2
            # should update -1.0 for [3, 4]
            [2, 0], [2, 0], [0, 2], [0, 2], [3, 2], [3, 2], [3, 4]
        ],
        [
            # 3 units at 1 confirmed negative, 4 unconfirmed, points gained = 1
            # should update 0.25 for [0, 0], [0, 4], [3, 3], [4, 4]
            [3, 2], [3, 2], [3, 2], [0, 0], [0, 4], [3, 3], [4, 3]
        ],
        [
            [2, 0], [2, 0], [2, 0], [0, 2], [0, 2], [-1, -1], [3, 4]
        ],
        [
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [-1, -1], [1, 1]
        ],
    ])
    points_gained2 = jnp.array([2, 1, 2, 2])

    expected_points_map2 = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.5],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.3333,  0.0],
            [  0.0,  0.0,  0.0, -1.0, -1.0]
        ],
        [
            [  0.25,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.25, 0.25],
            [  0.25,  0.0,  0.0,  1.0, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0, -1.0]
        ],
        [
            [  0.0,  1.0,  1.0,  0.0,  0.0],
            [  0.0,  1.0,  0.0,  0.0, -1.0],
            [  1.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0,  0.6667, -1.0]
        ],
    ])

    updated_points_map2 = update_points_map_batch(expected_points_map, mark_duplicates_batched(positions2), points_gained2)

    assert jnp.allclose(updated_points_map2, expected_points_map2)

def test_update_points_map_batch5():
    # conflicting
    points_map = jnp.array([
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0]
        ],
        [
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ]
    ])

    positions = jnp.array([
        [
            # 3 at confirmed positive, 2 at confirmed negative, 1 unconfirmed, points gained = 5
            # 2 points gained for 1 unconfirmed. This should not happen. But we need to cap it at 1.
            [2, 0], [2, 3], [3, 2], [2, 3], [-1, -1], [0, 0]
        ],
        [
            # 2 at confirmed positive, 1 unconfirmed, points gained = 0
            # 2 confirmed but no yield. Still we need to make [0, 0] confirmed negative
            [2, 0], [2, 3], [-1, -1], [-1, -1], [-1, -1], [0, 0]
        ],
    ])
    points_gained = jnp.array([5, 0])

    updated_points_map = update_points_map_batch(points_map, positions, points_gained)

    expected_points_map = jnp.array([
        [
            [  1.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0, -1.0]
        ],
        [
            [ -1.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
    ])
    assert jnp.allclose(updated_points_map, expected_points_map)


def test_filter_by_proximity():
    positions = jnp.array([
        [[1, 2], [3, 4], [10, 20], [5, 10], [3, 4]],
        [[17, 12], [7, 10], [15, 25], [11, 12], [13, 14]],
        [[1, 2], [3, 4], [10, 20], [5, 6], [3, 4]],
        [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[6, 9], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[0, 0], [1, 1], [1, 2], [0, 1], [2, 3]],
    ])
    relic_nodes = jnp.array([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[12, 22], [13, 23], [13, 12]],
        [[-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [4, 7]],
        [[-1, -1], [-1, -1], [-1, -1]],
    ])
    filtered_positions = filter_by_proximity_batch(positions, relic_nodes)
    print(filtered_positions)
    expected_positions = jnp.array([
        [
            [ 1,  2], [ 3,  4], [-1, -1], [ -1,  -1], [ 3,  4]
        ],
        [
            [ -1,  -1], [ 7, 10], [-1, -1], [11, 12], [13, 14]
        ],
        [
            [ -1,  -1], [ -1,  -1], [10, 20], [ -1,  -1], [ -1,  -1]
        ],
        [
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
        ],
        [
            [6, 9], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
        ],
        [
            [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
        ],
    ])
    
    assert jnp.array_equal(filtered_positions, expected_positions)


def test_update_points_map_with_relic_nodes():
    points_map = jnp.zeros((5, 5, 5))
    positions = jnp.array([
        [
            [0, 0], [1, 1],
        ],
        [
            [0, 1], [4, 3],
        ],
        [
            [2, 2], [4, 0],
        ],
        [
            [2, 2], [4, 0],
        ],
        [
            [-1, -1], [-1, -1],
        ],
    ])
    points_gained = jnp.array([3, 1, 2, 1, 5])   

    relic_nodes = jnp.array([
        [
            [4, 4], [4, 3] 
        ],
        [
            [0, 2], [0, 0], 
        ],
        [
            [2, 2], [4, 0],
        ],
        [
            [2, 2], [-1, -1],
        ],
        [
            [-1, -1], [-1, -1],
        ],
    ])

    updated_points_map = update_points_map_with_relic_nodes(
        points_map=points_map,
        relic_nodes=relic_nodes,
        positions=positions,
        points_gained=points_gained,
    )

    expected_points_map = jnp.array([
        [
            # should not update anything.
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            # only [0, 1] is in proximity
            # mirror is [3, 4]
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  1.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0,  0.0]
        ],
        [
            # two points in diagonal
            [  0.0,  0.0,  0.0,  0.0,  1.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            # two points in diagonal
            [  0.0,  0.0,  0.0,  0.0,  0.5],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.5,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
        [
            # should not update anything.
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  0.0,  0.0]
        ],
    ])

    assert jnp.array_equal(updated_points_map, expected_points_map)

def test_vmap_update_points_map_with_relic_nodes():
    points_map = jnp.zeros((1, 5, 5))
    points_history_positions = jnp.array([
        [
            [
                [-1, -1], [-1, -1], [-1, -1],
            ],
        ],
        [
            [
                [0, 1], [2, 2], [2, 2],
            ],
        ],
        [
            [
                [2, 2], [2, 2], [1, 1],
            ]
        ],
        [
            [
                [-1, -1], [-1, -1], [-1, -1],
            ]
        ],
    ])

    points_history = jnp.array([
        [
            [1]
        ],
        [
            [1]
        ],
        [
            [0]
        ],
        [
            [3]
        ]
    ])

    relic_nodes = jnp.array([
        [
            [-1, -1], [0, 0],
        ]
    ])

    updated_points_map = update_points_map_with_relic_nodes_scan(
        points_map,
        relic_nodes,
        points_history_positions,
        points_history
    )
    expected_points_map = jnp.array([
        [
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  0.5, -1.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0,  0.5,  0.0]
        ],
    ])

    assert jnp.array_equal(updated_points_map, expected_points_map)

    updated_points_map2 = update_points_map_with_relic_nodes_scan(
        updated_points_map,
        relic_nodes,
        points_history_positions,
        points_history
    )
    expected_points_map2 = jnp.array([
        [
            [  0.0,  0.0,  0.0,  0.0,  0.0],
            [  1.0, -1.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0, -1.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0,  0.0]
        ],
    ])

    assert jnp.array_equal(updated_points_map2, expected_points_map2)
