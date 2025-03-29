# readmetest

<details>
<summary>look_at — Create a transformation that looks at a target point.</summary>

<details>
<summary>
   
```py
def look_at(
    *,
    eye: Vector,
    look_axis: Vector,
    look_at: Optional[Vector] = None,
    look_along: Optional[Vector] = None,
    up_axis: Vector,
    up: Vector,
) -> "Trafo"
```

</summary>

```py
        eye: location
        look_axis: local view direction (from eye towards target)
        look_at: target if target is a point
        look_along: target if target is a direction
        up_axis: local up direction
        up: global up direction
        """

        # rotate the user-defined camera coordinate system (look_axis, up_axis)
        # to align with our convention:
        # x is the look axis (from eye towards target)
        # y is the up axis
        # z is the right axis

        if look_axis.norm() == 0:
            raise ValueError("look_axis must not be a zero vector")
        if up_axis.norm() == 0:
            raise ValueError("up_axis must not be a zero vector")
        if look_axis.cross(up_axis).norm() == 0:
            raise ValueError(
                "look_axis and up_axis must not be parallel or anti-parallel"
            )

        look_axis, up_axis, right_axis = Vector.make_basis(look_axis, up_axis)

        intrinsic_rotation = Rotation.from_matrix(
            [look_axis, up_axis, right_axis], check_matrix=False
        )

        # align our convention coordinate system with the environment (eye, target, up):
        # camera is located at eye
        # x points towards the target
        # roll until y points towards up as well as possible
        # z is determined from cross product

        if look_at is not None and look_along is None:
            look_along = look_at - eye
        elif look_at is None and look_along is not None:
            pass
        else:
            raise ValueError("either look_at or look_along must be set")

        if look_along.norm() == 0:
            # eye is located at target or target is the zero vector
            # orientation is undefined (but always kind of correct), return identity
            return Trafo(t=eye, r=Rotation.identity())
        if up.cross(look_along).norm() == 0:
            # up is parallel or anti-parallel to look direction or up is the zero vector
            # roll is undefined, just rotate the identity orientation to look towards the target
            return Trafo(t=eye).rotated_towards(look_axis, point_along=look_along)

        view_direction, up_direction, right_direction = Vector.make_basis(
            look_along, up
        )

        extrinsic_rotation = Rotation.from_matrix(
            [view_direction, up_direction, right_direction],
            row_major=False,
            check_matrix=False,
        )

        return Trafo(t=eye, r=extrinsic_rotation @ intrinsic_rotation)
```

</details>

</details>
