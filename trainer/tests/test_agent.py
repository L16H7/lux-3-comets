import sys
sys.path.append('./trainer')

import pytest
import jax.numpy as jnp

from agent import vectorized_transform_actions

def test_vectorized_transform_actions():
    actions = jnp.array([[1, 0, 2, 4, 3, 5, 3, 2, 0, 1]])

    expected_actions = jnp.array([[2, 0, 1, 3, 4, 5, 4, 1, 0, 2]])

    transformed_actions = vectorized_transform_actions(actions)
    print(transformed_actions)

    assert jnp.array_equal(transformed_actions, expected_actions)

def test_vectorized_transform_actions2():
    actions = jnp.array([[[1, 0, 2, 4, 3], [5, 3, 2, 0, 1]]])

    expected_actions = jnp.array([[[2, 0, 1, 3, 4], [5, 4, 1, 0, 2]]])

    transformed_actions = vectorized_transform_actions(actions)
    print(transformed_actions)

    assert jnp.array_equal(transformed_actions, expected_actions)
