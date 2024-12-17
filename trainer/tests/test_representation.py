import sys
sys.path.append('./trainer')

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from representation import create_relic_nodes_maps
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
