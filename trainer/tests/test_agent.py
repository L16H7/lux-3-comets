import sys
sys.path.append('./trainer')

import pytest
import jax.numpy as jnp

from agent import vectorized_transform_actions, mask_sap_range_vmap, mask_out_of_bounds, generate_attack_masks, generate_attack_masks_batch

def test_vectorized_transform_actions():
    actions = jnp.array([[1, 0, 2, 4, 3, 5, 3, 2, 0, 1]])

    expected_actions = jnp.array([[2, 0, 1, 3, 4, 5, 4, 1, 0, 2]])

    transformed_actions = vectorized_transform_actions(actions)

    assert jnp.array_equal(transformed_actions, expected_actions)

def test_vectorized_transform_actions2():
    actions = jnp.array([[[1, 0, 2, 4, 3], [5, 3, 2, 0, 1]]])

    expected_actions = jnp.array([[[2, 0, 1, 3, 4], [5, 4, 1, 0, 2]]])

    transformed_actions = vectorized_transform_actions(actions)

    assert jnp.array_equal(transformed_actions, expected_actions)

def test_mask_sap_range_vmap():
    sap_range_mask = jnp.ones((4, 3, 17))
    cutoff_range = jnp.array([0, 2, 3, 5])
    sap_range_masked = mask_sap_range_vmap(sap_range_mask, cutoff_range)

    expected_masked = jnp.array([
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
    ])

    assert jnp.array_equal(sap_range_masked, expected_masked)

def test_mask_out_of_bounds():
    agent_positions = jnp.array([
        [
            [1, 2],
            [4, 6],
        ],
        [
            [9, 10],
            [12, 14],
        ],
        [
            [22, 20],
            [16, 18],
        ]
    ])

    target_x, target_y = mask_out_of_bounds(agent_positions)

    print(target_x)
    expected_target_x = jnp.array([
        [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True],
        [False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    ])

    expected_target_y = jnp.array([
        [False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True],
        [False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False],
    ]) 
    
    assert jnp.array_equal(target_x, expected_target_x)
    assert jnp.array_equal(target_y, expected_target_y)
    

def test_generate_attack_masks():
    agent_positions = jnp.array([
        [20,  5], [ 6,  3], [11,  2], [ 2,  2], [20, 16], [12, 23], [17,  3], [13, 20]
    ])
    target_positions = jnp.array([
        [20,  6], [ 2,  8], [10, 16], [23, 14]
    ])

    expected_attack_mask_x = jnp.array([
        [False, False, False, False, False, False, False, False,  True,
            False, False,  False, False, False, False, False, False],
        [False, False, False, False,  True, False, False, False, False,
            False, False, False,  False, False, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
        False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False,  True,
            False, False, False, False, False, False, False,  False],
        [False, False, False, False, False, False, False, False,  False,
            False, False,  True, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False,
            False, False, False, False, False, False, False,  False],
        [False,  False, False, False, False, False, False, False, False,
            False, False,  True, False, False,  False, False, False],
        [False, False, False, False, False,  True, False, False, False,
            False, False, False, False, False, False,  False, False]
    ])
    expected_attack_mask_y = jnp.array([
        [False, False, False, False, False, False, False, False,  False,
            True, False,  False, False, False, False, False, False],
        [False, False, False, False,  False, False, False, False, False,
            False, False, False,  False, True, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
            False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False,  False,
            False, False, False, False, False, True, False,  False],
        [False, False, False, False, False, False, True, False,  False,
            False, False,  False, False, False, False, False, False],
        [False, True, False, False, False, False,  False, False, False,
            False, False, False, False, False, False, False,  False],
        [False,  False, False, False, False, False, False, False, False,
            False, False,  True, False, False,  False, False, False],
        [False, False, False, False, True,  False, False, False, False,
            False, False, False, False, False, False,  False, False]
    ])
    attack_mask_x, attack_mask_y = generate_attack_masks(
        agent_positions=agent_positions,
        target_positions=target_positions
    )
    print(attack_mask_y)
    assert jnp.array_equal(attack_mask_x, expected_attack_mask_x)
    assert jnp.array_equal(attack_mask_y, expected_attack_mask_y)

def test_generate_attack_masks2():
    agent_positions = jnp.array([
        [20,  5], [ -1,  -1], [23,  19], [2, 2]
    ])
    target_positions = jnp.array([
        [-1,  -1], [ 18,  20], [21, 9],
    ])
    expected_attack_mask_x = jnp.array([
        [False, False, False, False, False, False, False, False,  False,
            True, False,  False, False, False, False, False, False],
        [False, False, False, False,  False, False, False, False, False,
            False, False, False,  False, False, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
            False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
            False, False, False, False, False, False, False, False],
    ])
    expected_attack_mask_y = jnp.array([
        [False, False, False, False, False, False, False, False,  False,
            False, False,  False, True, False, False, False, False],
        [False, False, False, False,  False, False, False, False, False,
            False, False, False,  False, False, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
            False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False,  False, False,
            False, False, False, False, False, False, False, False],
    ])

    attack_mask_x, attack_mask_y = generate_attack_masks(
        agent_positions=agent_positions,
        target_positions=target_positions,
        x_range=4,
        y_range=4
    )
    assert jnp.array_equal(attack_mask_x, expected_attack_mask_x)
    assert jnp.array_equal(attack_mask_y, expected_attack_mask_y)

def test_generate_attack_masks_batch():
    agent_positions = jnp.array([
        [
            [20,  5], [ -1,  -1], [23,  19], [2, 2]
        ],
        [
            [0,  0], [ 3,  2], [5,  6], [0, 0]
        ],
    ])
    target_positions = jnp.array([
        [
            [-1,  -1], [ 18,  20], [21, 9],
        ],
        [
            [-1,  -1], [ 3,  5], [2, 2],
        ],
    ])
    expected_attack_masks = jnp.array([
        [
            [False, False, False, False, False, False, False, False,  False,
                True, False,  False, False, False, False, False, False],
            [False, False, False, False,  False, False, False, False, False,
                False, False, False,  False, False, False, False, False],
            [False, False, False, False, False, False, False,  False, False,
                False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False,  False, False,
                False, False, False, False, False, False, False, False],
        ],
        [
            [False, False, False, False, False, False, False, False,  False,
                False, True,  False, False, False, False, False, False],
            [False, False, False, False,  False, False, False, True, False,
                False, False, False,  False, False, False, False, False],
            [False, False, False, False, False, False, True,  False, False,
                False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False,  False, False,
                False, True, False, False, False, False, False, False],
        ],
    ])

    attack_masks_x, _ = generate_attack_masks_batch(
        agent_positions,
        target_positions,
        jnp.array([4, 2]),
        jnp.array([4, 2])
    )

    assert jnp.array_equal(attack_masks_x, expected_attack_masks)
    