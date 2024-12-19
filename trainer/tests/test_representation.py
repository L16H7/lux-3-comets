import sys
sys.path.append('./trainer')

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from representation import create_relic_nodes_maps, create_unit_maps, transform_coordinates, reconcile_positions
from constants import Constants

@pytest.fixture
def mock_constants(monkeypatch):
    # Mock Constants so that MAP_HEIGHT and MAP_WIDTH are controlled by the test
    monkeypatch.setattr(Constants, "MAP_HEIGHT", 5)
    monkeypatch.setattr(Constants, "MAP_WIDTH", 5)

def test_create_relic_nodes_maps(mock_constants):
    # Arrange
    # Suppose we have 2 environments (n_envs=2), each with 2 relic nodes (n_relic_nodes=2).
    # relic_nodes shape: (n_envs, n_relic_nodes, 2)
    # Each relic node is (x, y).
    # For simplicity, let's place the nodes at (1,1) and (2,3) for the first env
    # and (0,0) and (4,4) for the second env.
    # This means:
    # Env 1: relic nodes => [(1,1), (2,3)]
    # Env 2: relic nodes => [(0,0), (4,4)]
    relic_nodes = jnp.array([
        [[1, 2],
         [2, 3]],  # For environment 0
        [[0, 1],
         [4, 3]]   # For environment 1
    ])

    # Act
    relic_nodes_maps = create_relic_nodes_maps(relic_nodes)

    expected_maps = jnp.zeros((2, 5, 5), dtype=jnp.int32)
    expected_maps = expected_maps.at[0, 2, 1].set(1)
    expected_maps = expected_maps.at[0, 3, 2].set(1)
    expected_maps = expected_maps.at[1, 1, 0].set(1)
    expected_maps = expected_maps.at[1, 3, 4].set(1)

    # Act
    relic_nodes_maps = create_relic_nodes_maps(relic_nodes)

    # Assert
    # Check the shape first
    assert relic_nodes_maps.shape == (2, 5, 5)

    # Now check that the relic nodes maps are exactly as expected
    assert jnp.array_equal(relic_nodes_maps, expected_maps)

    # Alternatively, for a more detailed failure message, you could use numpy.testing
    np.testing.assert_array_equal(relic_nodes_maps, expected_maps)

@pytest.fixture
def mock_constants2(monkeypatch):
    # Mock Constants so that MAP_HEIGHT and MAP_WIDTH are controlled by the test
    monkeypatch.setattr(Constants, "MAP_HEIGHT", 10)
    monkeypatch.setattr(Constants, "MAP_WIDTH", 10)

def test_create_relic_nodes_maps2(mock_constants2):
    # Arrange
    # Suppose we have 2 environments (n_envs=2), each with 2 relic nodes (n_relic_nodes=2).
    # relic_nodes shape: (n_envs, n_relic_nodes, 2)
    # Each relic node is (x, y).
    # For simplicity, let's place the nodes at (1,1) and (2,3) for the first env
    # and (0,0) and (4,4) for the second env.
    # This means:
    # Env 1: relic nodes => [(1,1), (2,3)]
    # Env 2: relic nodes => [(0,0), (4,4)]
    relic_nodes = jnp.array([
        [[1, 2],
         [5, 3],
         [4, 1]],  # For environment 0
        [[9, 1],
         [8, 7],
         [5, 9]]   # For environment 1
    ])

    # Act
    relic_nodes_maps = create_relic_nodes_maps(relic_nodes)

    expected_maps = jnp.zeros((2, 10, 10), dtype=jnp.int32)
    expected_maps = expected_maps.at[0, 2, 1].set(1)
    expected_maps = expected_maps.at[0, 3, 5].set(1)
    expected_maps = expected_maps.at[0, 1, 4].set(1)
    expected_maps = expected_maps.at[1, 1, 9].set(1)
    expected_maps = expected_maps.at[1, 7, 8].set(1)
    expected_maps = expected_maps.at[1, 9, 5].set(1)

    # Act
    relic_nodes_maps = create_relic_nodes_maps(relic_nodes)

    # Assert
    # Check the shape first
    assert relic_nodes_maps.shape == (2, 10, 10)

    # Now check that the relic nodes maps are exactly as expected
    assert jnp.array_equal(relic_nodes_maps, expected_maps)

    # Alternatively, for a more detailed failure message, you could use numpy.testing
    np.testing.assert_array_equal(relic_nodes_maps, expected_maps)

def test_create_unit_maps(mock_constants2):
    # Arrange
    n_envs = 2
    n_units = 3

    # Define unit positions as [[[x1, y1], [x2, y2], [x3, y3]], [[x1, y1], [x2, y2], [x3, y3]]]
    unit_positions = jnp.array([
        [[1, 2], [2, 3], [5, 5]],
        [[0, 2], [4, 4], [9, 4]]
    ])
    # Define unit masks, 1 where unit exists, 0 otherwise
    unit_masks = jnp.array([
        [1, 1, 0],
        [1, 0, 1]
    ])
    # Define unit energy
    unit_energy = jnp.array([
        [100, 150, 0],
        [200, 0, 300]
    ])

    # Expected unit maps and energy maps based on above inputs
    expected_unit_maps = jnp.zeros((n_envs, 10, 10), dtype=jnp.int32)
    expected_unit_maps = expected_unit_maps.at[0, 2, 1].set(1)
    expected_unit_maps = expected_unit_maps.at[0, 3, 2].set(1)
    expected_unit_maps = expected_unit_maps.at[1, 2, 0].set(1)
    expected_unit_maps = expected_unit_maps.at[1, 4, 9].set(1)

    expected_unit_energy_maps = jnp.zeros((n_envs, 10, 10), dtype=jnp.float32)
    expected_unit_energy_maps = expected_unit_energy_maps.at[0, 2, 1].set(100)
    expected_unit_energy_maps = expected_unit_energy_maps.at[0, 3, 2].set(150)
    expected_unit_energy_maps = expected_unit_energy_maps.at[1, 2, 0].set(200)
    expected_unit_energy_maps = expected_unit_energy_maps.at[1, 4, 9].set(300)

    # Act
    unit_maps, unit_energy_maps = create_unit_maps(unit_positions, unit_masks, unit_energy)

    # Assert
    # Check the shape first
    assert unit_maps.shape == (n_envs, 10, 10)
    assert unit_energy_maps.shape == (n_envs, 10, 10)

    # Now check that the unit maps and energy maps are exactly as expected
    assert jnp.all(unit_maps == expected_unit_maps)
    assert jnp.all(unit_energy_maps == expected_unit_energy_maps)

def test_create_unit_maps2(mock_constants2):
    # Arrange
    n_envs = 2
    n_units = 3

    # Define unit positions as [[[x1, y1], [x2, y2], [x3, y3]], [[x1, y1], [x2, y2], [x3, y3]]]
    unit_positions = jnp.array([
        [[1, 2], [2, 3], [5, 5], [4, 3]],
        [[0, 2], [4, 4], [9, 4], [0, 2]]
    ])
    # Define unit masks, 1 where unit exists, 0 otherwise
    unit_masks = jnp.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1]
    ])
    # Define unit energy
    unit_energy = jnp.array([
        [100, 150, 0, 3],
        [200, 0, 300, 4]
    ])

    # Expected unit maps and energy maps based on above inputs
    expected_unit_maps = jnp.zeros((n_envs, 10, 10), dtype=jnp.int32)
    expected_unit_maps = expected_unit_maps.at[0, 2, 1].set(1)
    expected_unit_maps = expected_unit_maps.at[0, 3, 2].set(1)
    expected_unit_maps = expected_unit_maps.at[0, 3, 4].set(1)
    expected_unit_maps = expected_unit_maps.at[1, 2, 0].set(2)
    expected_unit_maps = expected_unit_maps.at[1, 4, 9].set(1)

    expected_unit_energy_maps = jnp.zeros((n_envs, 10, 10), dtype=jnp.float32)
    expected_unit_energy_maps = expected_unit_energy_maps.at[0, 2, 1].set(100)
    expected_unit_energy_maps = expected_unit_energy_maps.at[0, 3, 2].set(150)
    expected_unit_energy_maps = expected_unit_energy_maps.at[0, 3, 4].set(3)
    expected_unit_energy_maps = expected_unit_energy_maps.at[1, 2, 0].set(204)
    expected_unit_energy_maps = expected_unit_energy_maps.at[1, 4, 9].set(300)

    # Act
    unit_maps, unit_energy_maps = create_unit_maps(unit_positions, unit_masks, unit_energy)

    # Assert
    # Check the shape first
    assert unit_maps.shape == (n_envs, 10, 10)
    assert unit_energy_maps.shape == (n_envs, 10, 10)

    # Now check that the unit maps and energy maps are exactly as expected
    assert jnp.all(unit_maps == expected_unit_maps)
    assert jnp.all(unit_energy_maps == expected_unit_energy_maps)

def test_transform_coordinates():
    input_positions = jnp.array([[[0, 0], [23, 23], [12, 0], [0, 23]],
                                [[23, 0], [0, 23], [12, 23], [23, 0]],
                                [[11, 11], [12, 12], [13, 13], [14, 14]],
                                [[0, 12], [23, 12], [12, 20], [12, 0]]])

    # Expected output after horizontal flip and 90-degree rotation clockwise
    expected_output = jnp.array([[[23, 23], [0, 0], [23, 11], [0, 23]],
                                    [[23, 0], [0, 23], [0, 11], [23, 0]],
                                    [[12, 12], [11, 11], [10, 10], [9, 9]],
                                    [[11, 23], [11, 0], [3, 11], [23, 11]]])

    transformed_positions = transform_coordinates(input_positions)

    assert jnp.array_equal(transformed_positions, expected_output)

def test_transform_coordinates2():
    input_positions = jnp.array([[[[0, 0], [23, 23], [12, 0], [0, 23]],
                                [[23, 0], [0, 23], [12, 23], [23, 0]],
                                [[11, 11], [12, 12], [13, 13], [14, 14]],
                                [[0, 12], [23, 12], [12, 20], [12, 0]]]])

    # Expected output after horizontal flip and 90-degree rotation clockwise
    expected_output = jnp.array([[[[23, 23], [0, 0], [23, 11], [0, 23]],
                                    [[23, 0], [0, 23], [0, 11], [23, 0]],
                                    [[12, 12], [11, 11], [10, 10], [9, 9]],
                                    [[11, 23], [11, 0], [3, 11], [23, 11]]]])

    transformed_positions = transform_coordinates(input_positions)

    assert jnp.array_equal(transformed_positions, expected_output)

def test_transform_coordinates3():
    input_positions = jnp.array([[[[0, 0], [23, 23], [12, 0], [0, 23]],
                                [[23, 0], [0, 23], [12, 23], [23, 0]]],
                                [[[11, 11], [12, 12], [13, 13], [14, 14]],
                                [[0, 12], [23, 12], [12, 20], [12, 0]]]])

    # Expected output after horizontal flip and 90-degree rotation clockwise
    expected_output = jnp.array([[[[23, 23], [0, 0], [23, 11], [0, 23]],
                                    [[23, 0], [0, 23], [0, 11], [23, 0]]],
                                    [[[12, 12], [11, 11], [10, 10], [9, 9]],
                                    [[11, 23], [11, 0], [3, 11], [23, 11]]]])

    transformed_positions = transform_coordinates(input_positions)

    assert jnp.array_equal(transformed_positions, expected_output)

@pytest.fixture
def mock_constants24(monkeypatch):
    # Mock Constants so that MAP_HEIGHT and MAP_WIDTH are controlled by the test
    monkeypatch.setattr(Constants, "MAP_HEIGHT", 24)
    monkeypatch.setattr(Constants, "MAP_WIDTH", 24)


def test_reconcile_positions(mock_constants24):
    positions = jnp.array([
        [[3, 2], [-1, -1], [2, 1], [-1, -1], [23, 14], [-1, -1]],
        [[0, 0], [11, 11], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [12, 23], [-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1], [21, 21], [-1, -1], [-1, -1]],
    ])
    reconciled_positions = reconcile_positions(positions)
    expected_positions = jnp.array([
        [[3,  2], [9, 0], [2, 1], [21, 20], [23, 14], [22, 21]],
        [[0,  0], [11, 11], [-1, -1], [23, 23], [12, 12], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [12, 23], [-1, -1], [-1, -1], [0, 11]],
        [[2, 2], [-1, -1], [-1, -1], [21, 21], [-1, -1], [-1, -1]],
    ])

    assert jnp.array_equal(reconciled_positions, expected_positions)
