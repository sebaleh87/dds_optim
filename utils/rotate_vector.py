import jax.numpy as jnp

def rotate_vector(vector, angle, axis=None):
    """
    Rotates a 2D or 3D vector about the origin.
    
    Args:
        vector: jnp.array, shape (2,) for 2D or (3,) for 3D.
        angle: float, angle of rotation in radians.
        axis: jnp.array or None, the axis of rotation (used for 3D vectors).
              Should be shape (3,). Ignored for 2D vectors.
    
    Returns:
        Rotated vector as a jnp.array.
    """
    if vector.shape[0] == 2:
        # 2D rotation matrix
        rotation_matrix = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle),  jnp.cos(angle)]
        ])
        return jnp.dot(rotation_matrix, vector)
    
    elif vector.shape[0] == 3:
        if axis is None:
            raise ValueError("Axis must be provided for 3D rotation.")
        axis = axis / jnp.linalg.norm(axis)  # Normalize axis
        ux, uy, uz = axis
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        one_c = 1 - c
        
        # 3D rotation matrix using the Rodrigues' rotation formula
        rotation_matrix = jnp.array([
            [c + ux**2 * one_c, ux*uy*one_c - uz*s, ux*uz*one_c + uy*s],
            [uy*ux*one_c + uz*s, c + uy**2 * one_c, uy*uz*one_c - ux*s],
            [uz*ux*one_c - uy*s, uz*uy*one_c + ux*s, c + uz**2 * one_c]
        ])
        return jnp.dot(rotation_matrix, vector)
    
    else:
        raise ValueError("Vector must be either 2D or 3D.")

if(__name__ == "__main__"):
    # Example usage:
    vector_2d = jnp.array([1.0, 0.0])
    angle = jnp.pi /2  # 45 degrees in radians
    rotated_2d = rotate_vector(vector_2d, angle)
    print("Rotated 2D Vector:", rotated_2d)

    vector_3d = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.array([0.0, 0.0, 1.0])  # Rotate around the z-axis
    rotated_3d = rotate_vector(vector_3d, angle, axis)
    print("Rotated 3D Vector:", rotated_3d)
