<details><summary><b><code>class Vector</code></b> — An immutable 3D vector representing a point or a direction.</summary>
<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>zero</code> — Create a zero vector.</summary>
<details><summary>

```py
    def zero() -> "Vector":
```

</summary>

```py
"""Create a zero vector."""
        return Vector(0.0, 0.0, 0.0)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>ex</code> — Create a unit vector in x-direction.</summary>
<details><summary>

```py
    def ex() -> "Vector":
```

</summary>

```py
"""Create a unit vector in x-direction."""
        return Vector(1.0, 0.0, 0.0)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>ey</code> — Create a unit vector in y-direction.</summary>
<details><summary>

```py
    def ey() -> "Vector":
```

</summary>

```py
"""Create a unit vector in y-direction."""
        return Vector(0.0, 1.0, 0.0)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>ez</code> — Create a unit vector in z-direction.</summary>
<details><summary>

```py
    def ez() -> "Vector":
```

</summary>

```py
"""Create a unit vector in z-direction."""
        return Vector(0.0, 0.0, 1.0)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rand_box</code> — Create a random vector with uniform distribution within a box.</summary>
<details><summary>

```py
    def rand_box(
        min: Sequence[float] = (0.0, 0.0, 0.0),
        max: Sequence[float] = (1.0, 1.0, 1.0),
        generator: random.Random = cast(random.Random, random),
    ) -> "Vector":
```

</summary>

```py
"""Create a random vector with uniform distribution within a box."""
        return Vector(
            generator.uniform(min[0], max[0]),
            generator.uniform(min[1], max[1]),
            generator.uniform(min[2], max[2]),
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rand_sphere</code> — Create a random vector with uniform distribution on or in a sphere.</summary>
<details><summary>

```py
    def rand_sphere(
        radius: float = 1.0,
        center: Optional["Vector"] = None,
        fill: bool = False,
        generator: random.Random = cast(random.Random, random),
    ) -> "Vector":
```

</summary>

```py
"""Create a random vector with uniform distribution on or in a sphere."""
        v = Vector(generator.gauss(), generator.gauss(), generator.gauss()).normalized()
        if center is None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__eq__</code> — Check equality with another vector; true if all elements are equal.</summary>
<details><summary>

```py
    def __eq__(self, other: object) -> bool:
```

</summary>

```py
"""Check equality with another vector; true if all elements are equal."""
        if isinstance(other, Vector)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__ne__</code> — Check inequality with another vector; true if any element is unequal.</summary>
<details><summary>

```py
    def __ne__(self, other: object) -> bool:
```

</summary>

```py
"""Check inequality with another vector; true if any element is unequal."""
        return not self == other

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__add__</code> — Add another vector element-wise.
Note that this violates LSP for tuples which are expected to concatenate instead.</summary>
<details><summary>

```py
    def __add__(self, other: "Vector") -> "Vector":  # type: ignore[override]
        """Add another vector element-wise.
        Note that this violates LSP for tuples which are expected to concatenate instead.
        """
        if isinstance(other, Vector):
```

</summary>

```py
    return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        else
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__neg__</code> — Return the negated vector.</summary>
<details><summary>

```py
    def __neg__(self) -> "Vector":
```

</summary>

```py
"""Return the negated vector."""
        return Vector(-self.x, -self.y, -self.z)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__sub__</code> — Subtract another vector.</summary>
<details><summary>

```py
    def __sub__(self, other: "Vector") -> "Vector":
```

</summary>

```py
"""Subtract another vector."""
        if isinstance(other, Vector)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__mul__</code> — Multiply by a scalar element-wise.
Note that this violates LSP for tuples which are expected to repeat instead.</summary>
<details><summary>

```py
    def __mul__(self, scalar: float) -> "Vector":  # type: ignore[override]
        """Multiply by a scalar element-wise.
        Note that this violates LSP for tuples which are expected to repeat instead."""
        if isinstance(scalar, (int, float)):
```

</summary>

```py
    return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
        else
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__rmul__</code> — Multiply by a scalar element-wise.
Note that this violates LSP for tuples which are expected to repeat instead.</summary>
<details><summary>

```py
    def __rmul__(self, scalar: float) -> "Vector":  # type: ignore[override]
        """Multiply by a scalar element-wise.
        Note that this violates LSP for tuples which are expected to repeat instead."""
        return self * scalar
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__truediv__</code> — Divide by a scalar element-wise.</summary>
<details><summary>

```py
    def __truediv__(self, scalar: float) -> "Vector":
```

</summary>

```py
"""Divide by a scalar element-wise."""
        if isinstance(scalar, (int, float))
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>dot</code> — Calculate the dot product with another vector.</summary>
<details><summary>

```py
    def dot(self, other: "Vector") -> float:
```

</summary>

```py
"""Calculate the dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>cross</code> — Calculate the cross product with another vector.</summary>
<details><summary>

```py
    def cross(self, other: "Vector") -> "Vector":
```

</summary>

```py
"""Calculate the cross product with another vector."""
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>norm</code> — Calculate the Euclidean norm (length) of the vector.</summary>
<details><summary>

```py
    def norm(self) -> float:
```

</summary>

```py
"""Calculate the Euclidean norm (length) of the vector."""
        return norm(*self)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>length</code> — Calculate the length (Euclidean norm) of the vector.</summary>
<details><summary>

```py
    def length(self) -> float:
```

</summary>

```py
"""Calculate the length (Euclidean norm) of the vector."""
        return norm(*self)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>normalized</code> — Return a normalized (unit) vector with the same direction; raises when called on a zero vector.</summary>
<details><summary>

```py
    def normalized(self) -> "Vector":
```

</summary>

```py
"""Return a normalized (unit) vector with the same direction; raises when called on a zero vector."""
        x, y, z = self.x, self.y, self.z
        m = max(abs(x), abs(y), abs(z))
        if m == 0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>perp</code> — Calculate a vector perpendicular to this vector;
if other is given, the result is perpendicular to both.
Raises when called on a zero vector or when the vectors are parallel.</summary>
<details><summary>

```py
    def perp(self, other: Optional["Vector"] = None) -> "Vector":
```

</summary>

```py
"""Calculate a vector perpendicular to this vector;
        if other is given, the result is perpendicular to both.
        Raises when called on a zero vector or when the vectors are parallel."""
        if other is None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>make_basis</code> — Create an orthonormal basis from two vectors;
the direction of the first vector is preserved,
the second is made perpendicular to the first,
the third is perpendicular to the first two.
Raises when called on a zero vector or when the vectors are parallel.</summary>
<details><summary>

```py
    def make_basis(v1: "Vector", v2: "Vector") -> Tuple["Vector", "Vector", "Vector"]:
```

</summary>

```py
"""Create an orthonormal basis from two vectors;
        the direction of the first vector is preserved,
        the second is made perpendicular to the first,
        the third is perpendicular to the first two.
        Raises when called on a zero vector or when the vectors are parallel."""
        v1 = v1.normalized()
        v3 = v1.perp(v2)
        v2 = v3.perp(v1)
        return v1, v2, v3

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>distance_to</code> — Calculate the Euclidean distance to another vector.</summary>
<details><summary>

```py
    def distance_to(self, other: "Vector") -> float:
```

</summary>

```py
"""Calculate the Euclidean distance to another vector."""
        return (self - other).norm()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>angle_to</code> — Calculate the angle to another vector.</summary>
<details><summary>

```py
    def angle_to(self, other: "Vector") -> float:
```

</summary>

```py
"""Calculate the angle to another vector."""
        v1 = self.normalized()
        v2 = other.normalized()
        cos_angle = v1.dot(v2)
        sin_angle = v1.cross(v2).norm()
        return atan2(sin_angle, cos_angle)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>lerp</code> — Linearly interpolate between two vectors.</summary>
<details><summary>

```py
    def lerp(self, other: "Vector", f: float) -> "Vector":
```

</summary>

```py
"""Linearly interpolate between two vectors."""
        return Vector(
            self.x + f * (other.x - self.x),
            self.y + f * (other.y - self.y),
            self.z + f * (other.z - self.z),
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>mean</code> — Calculate the weighted mean of a sequence of vectors.
Raises when called with an empty sequence or when the sum of weights is zero.</summary>
<details><summary>

```py
    def mean(
        vectors: Iterable["Vector"], weights: Optional[Iterable[float]] = None
    ) -> "Vector":
```

</summary>

```py
"""Calculate the weighted mean of a sequence of vectors.
        Raises when called with an empty sequence or when the sum of weights is zero."""

        weights = weights or repeat(1.0)

        if (
            hasattr(vectors, "__len__")
            and hasattr(weights, "__len__")
            and vectors.__len__() != weights.__len__()
        )
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__str__</code> — Return a string representation of the vector.</summary>
<details><summary>

```py
    def __str__(self) -> str:
```

</summary>

```py
"""Return a string representation of the vector."""
        return f"(x={self.x}, y={self.y}, z={self.z})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__format__</code> — Return a formatted string representation of the vector;
the format_spec is applied to each element.</summary>
<details><summary>

```py
    def __format__(self, format_spec: str) -> str:
```

</summary>

```py
"""Return a formatted string representation of the vector;
        the format_spec is applied to each element."""
        fx = self.x.__format__(format_spec)
        fy = self.y.__format__(format_spec)
        fz = self.z.__format__(format_spec)
        return f"(x={fx}, y={fy}, z={fz})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__repr__</code> — Return an eval-able string representation of the vector.</summary>
<details><summary>

```py
    def __repr__(self) -> str:
```

</summary>

```py
"""Return an eval-able string representation of the vector."""
        return f"Vector({self.x}, {self.y}, {self.z})"

```

</details>
</details>

</details>

<details><summary><b><code>class Rotation</code></b> — An immutable 3D orientation or rotation.</summary>
<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__setattr__</code> — (no docstring)</summary>
<details><summary>

```py
    def __setattr__(self, name: str, value: float) -> None:
```

</summary>

```py
raise AttributeError("Rotation is immutable")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__delattr__</code> — (no docstring)</summary>
<details><summary>

```py
    def __delattr__(self, name: str) -> None:
```

</summary>

```py
raise AttributeError("Rotation is immutable")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__new__</code> — Construct a rotation from quaternion components without normalization,
intended only for use in classmethods. Use from_quat instead.</summary>
<details><summary>

```py
    def __new__(
        cls: Type[_T], *, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0
    ) -> _T:
```

</summary>

```py
"""Construct a rotation from quaternion components without normalization,
        intended only for use in classmethods. Use from_quat instead."""
        instance = super().__new__(cls)
        object.__setattr__(instance, "_x", x)
        object.__setattr__(instance, "_y", y)
        object.__setattr__(instance, "_z", z)
        object.__setattr__(instance, "_w", w)
        return instance

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__init__</code> — Create the identity rotation.</summary>
<details><summary>

```py
    def __init__(self) -> None:
```

</summary>

```py
"""Create the identity rotation."""
        # we only allow creation of the identity rotation via standard constructor
        # x, y, z, w are implementation details and should only be set via from_quat from the outside
        pass

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>identity</code> — Create the identity rotation while being extra explicit about it.</summary>
<details><summary>

```py
    def identity() -> "Rotation":
```

</summary>

```py
"""Create the identity rotation while being extra explicit about it."""
        return Rotation()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>x</code> — Create a rotation about the x-axis.</summary>
<details><summary>

```py
    def x(angle: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation about the x-axis."""
        return Rotation.__new__(Rotation, x=sin(angle / 2.0), w=cos(angle / 2.0))

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>y</code> — Create a rotation about the y-axis.</summary>
<details><summary>

```py
    def y(angle: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation about the y-axis."""
        return Rotation.__new__(Rotation, y=sin(angle / 2.0), w=cos(angle / 2.0))

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>z</code> — Create a rotation about the z-axis.</summary>
<details><summary>

```py
    def z(angle: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation about the z-axis."""
        return Rotation.__new__(Rotation, z=sin(angle / 2.0), w=cos(angle / 2.0))

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_quat</code> — Create a rotation from quaternion components.
Raises an error if the norm deviates from 1 beyond the specified tolerance
or if all components are 0.</summary>
<details><summary>

```py
    def from_quat(
        *, x: float, y: float, z: float, w: float, tolerance: float = EPS
    ) -> "Rotation":
```

</summary>

```py
"""Create a rotation from quaternion components.
        Raises an error if the norm deviates from 1 beyond the specified tolerance
        or if all components are 0."""

        m = max(abs(x), abs(y), abs(z), abs(w))
        if m == 0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_quat</code> — Return the quaternion components in the specified order.</summary>
<details><summary>

```py
    def as_quat(
        self, order: Literal["xyzw", "wxyz"]
    ) -> Tuple[float, float, float, float]:
```

</summary>

```py
"""Return the quaternion components in the specified order."""
        if order == "xyzw"
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_axis_angle</code> — Create a rotation from an axis and an angle.</summary>
<details><summary>

```py
    def from_axis_angle(axis: Vector, angle: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation from an axis and an angle."""
        axis = axis.normalized()
        half_angle = angle / 2.0
        sin_half_angle = sin(half_angle)
        return Rotation.__new__(
            Rotation,
            x=sin_half_angle * axis.x,
            y=sin_half_angle * axis.y,
            z=sin_half_angle * axis.z,
            w=cos(half_angle),
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_axis_angle</code> — Return the axis and angle of the rotation.
The angle is in the range [0, pi).
If the angle is 0, the axis is (1, 0, 0).</summary>
<details><summary>

```py
    def as_axis_angle(self) -> Tuple[Vector, float]:
```

</summary>

```py
"""Return the axis and angle of the rotation.
        The angle is in the range [0, pi).
        If the angle is 0, the axis is (1, 0, 0)."""
        x, y, z, w = self._x, self._y, self._z, self._w
        cos_half_angle = w
        sin_half_angle = norm(x, y, z)
        if sin_half_angle == 0.0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_rotvec</code> — Create a rotation from a rotation vector.
(The rotation vector is the axis of rotation scaled by the angle of rotation.)</summary>
<details><summary>

```py
    def from_rotvec(rotvec: Vector) -> "Rotation":
```

</summary>

```py
"""Create a rotation from a rotation vector.
        (The rotation vector is the axis of rotation scaled by the angle of rotation.)
        """
        angle = rotvec.norm()
        if angle == 0.0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_rotvec</code> — Return the rotation vector of the rotation.
(The rotation vector is the axis of rotation scaled by the angle of rotation.)</summary>
<details><summary>

```py
    def as_rotvec(self) -> Vector:
```

</summary>

```py
"""Return the rotation vector of the rotation.
        (The rotation vector is the axis of rotation scaled by the angle of rotation.)
        """
        axis, angle = self.as_axis_angle()
        return axis * angle

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_matrix</code> — Create a rotation from a 3x3 rotation matrix.</summary>
<details><summary>

```py
    def from_matrix(
        matrix: Sequence[Sequence[float]],
        row_major: bool = True,
        check_matrix: bool = True,
    ) -> "Rotation":
```

</summary>

```py
"""Create a rotation from a 3x3 rotation matrix."""
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

        if check_matrix
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_matrix</code> — Return the rotation as a 3x3 rotation matrix.</summary>
<details><summary>

```py
    def as_matrix(self, row_major: bool = True) -> List[List[float]]:
```

</summary>

```py
"""Return the rotation as a 3x3 rotation matrix."""
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        x, y, z, w = self._x, self._y, self._z, self._w
        xx, xy, xz, xw = x * x, x * y, x * z, x * w
        yy, yz, yw = y * y, y * z, y * w
        zz, zw = z * z, z * w

        if row_major
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>basis</code> — Return the basis vectors of the rotation.</summary>
<details><summary>

```py
    def basis(self) -> Tuple[Vector, Vector, Vector]:
```

</summary>

```py
"""Return the basis vectors of the rotation."""
        x, y, z = self.as_matrix(row_major=False)
        return Vector(*x), Vector(*y), Vector(*z)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>compose</code> — Compose a rotation from a sequence of rotations about x, y and z.
The sequence is an arbitrarily long string of axis identifiers, e.g. 'XY' or 'zyxZ'.
Use Capital letters for intrinsic rotations (rotate about the new, rotated axes),
use lowercase letters for extrinsic rotations (rotate about the world axes).
Intrinsic and extrinsic rotations can be mixed.</summary>
<details><summary>

```py
    def compose(sequence: str, angles: Sequence[float]) -> "Rotation":
```

</summary>

```py
"""Compose a rotation from a sequence of rotations about x, y and z.
        The sequence is an arbitrarily long string of axis identifiers, e.g. 'XY' or 'zyxZ'.
        Use Capital letters for intrinsic rotations (rotate about the new, rotated axes),
        use lowercase letters for extrinsic rotations (rotate about the world axes).
        Intrinsic and extrinsic rotations can be mixed."""
        if len(sequence) != len(angles)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_euler</code> — Create a rotation from Euler angles. The following orders are allowed:
ZXZ, XYX, YZY, ZYZ, XZX, YXY (proper Euler, intrinsic)
XYZ, YZX, ZXY, XZY, ZYX, YXZ (Tait-Bryan, intrinsic)
zxz, xyx, yzy, zyz, xzx, yxy (proper Euler, extrinsic)
xyz, yzx, zxy, xzy, zyx, yxz (Tait-Bryan, extrinsic)
intrinsic: rotate about the new, rotated axes
extrinsic: rotate about the original axes</summary>
<details><summary>

```py
    def from_euler(order: str, angles: Sequence[float]) -> "Rotation":
```

</summary>

```py
"""Create a rotation from Euler angles. The following orders are allowed
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_euler</code> — Return the Euler angles of the rotation. The order is one of:
ZXZ, XYX, YZY, ZYZ, XZX, YXY (proper Euler, intrinsic)
XYZ, YZX, ZXY, XZY, ZYX, YXZ (Tait-Bryan, intrinsic)
zxz, xyx, yzy, zyz, xzx, yxy (proper Euler, extrinsic)
xyz, yzx, zxy, xzy, zyx, yxz (Tait-Bryan, extrinsic)
intrinsic: rotate about the new, rotated axes
extrinsic: rotate about the original axes
In case of a singularity, the first angle is set to 0 for extrinsic rotations,
the third angle is set to 0 for intrinsic rotations.</summary>
<details><summary>

```py
    def as_euler(self, order: str) -> Tuple[float, float, float]:
```

</summary>

```py
"""Return the Euler angles of the rotation. The order is one of
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_ypr</code> — Create a rotation from yaw, pitch and roll angles.</summary>
<details><summary>

```py
    def from_ypr(yaw: float, pitch: float, roll: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation from yaw, pitch and roll angles."""
        return Rotation.compose("ZYX", [yaw, pitch, roll])

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_ypr</code> — Return the yaw, pitch and roll angles of the rotation.
In case of a singularity, roll is set to 0.</summary>
<details><summary>

```py
    def as_ypr(self) -> Tuple[float, float, float]:
```

</summary>

```py
"""Return the yaw, pitch and roll angles of the rotation.
        In case of a singularity, roll is set to 0."""
        return self.as_euler("ZYX")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_rpy</code> — Create a rotation from roll, pitch and yaw angles.</summary>
<details><summary>

```py
    def from_rpy(roll: float, pitch: float, yaw: float) -> "Rotation":
```

</summary>

```py
"""Create a rotation from roll, pitch and yaw angles."""
        return Rotation.compose("xyz", [roll, pitch, yaw])

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_rpy</code> — Return the roll, pitch and yaw angles of the rotation.
In case of a singularity, roll is set to 0.</summary>
<details><summary>

```py
    def as_rpy(self) -> Tuple[float, float, float]:
```

</summary>

```py
"""Return the roll, pitch and yaw angles of the rotation.
        In case of a singularity, roll is set to 0."""
        return self.as_euler("xyz")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rand</code> — Create a random rotation with uniform distribution.</summary>
<details><summary>

```py
    def rand(generator: random.Random = cast(random.Random, random)) -> "Rotation":
```

</summary>

```py
"""Create a random rotation with uniform distribution."""
        x = generator.gauss()
        y = generator.gauss()
        z = generator.gauss()
        w = generator.gauss()
        # should be impossible for Mersenne twister to generate a zero quaternion
        return Rotation.from_quat(x=x, y=y, z=z, w=w, tolerance=_inf)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>_rotate_vector</code> — Rotate a vector by the rotation.</summary>
<details><summary>

```py
    def _rotate_vector(self, other: Vector) -> Vector:
```

</summary>

```py
"""Rotate a vector by the rotation."""
        qx, qy, qz, qw = self._x, self._y, self._z, self._w
        px, py, pz = other  # pw = 0

        qpx = qw * px + qy * pz - qz * py
        qpy = qw * py - qx * pz + qz * px
        qpz = qw * pz + qx * py - qy * px
        qpw = -qx * px - qy * py - qz * pz

        qpq_x = -qpw * qx + qpx * qw - qpy * qz + qpz * qy
        qpq_y = -qpw * qy + qpx * qz + qpy * qw - qpz * qx
        qpq_z = -qpw * qz - qpx * qy + qpy * qx + qpz * qw

        return Vector(qpq_x, qpq_y, qpq_z)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>_rotate_quat</code> — Combine two rotations (quaternion multiplication).</summary>
<details><summary>

```py
    def _rotate_quat(self, other: "Rotation") -> Tuple[float, float, float, float]:
```

</summary>

```py
"""Combine two rotations (quaternion multiplication)."""
        x1, y1, z1, w1 = self._x, self._y, self._z, self._w
        x2, y2, z2, w2 = other._x, other._y, other._z, other._w
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        # does not return Rotation to let the caller decide whether to normalize
        return x, y, z, w

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Combine two rotations.</summary>
<details><summary>

```py
    def __matmul__(self, other: "Rotation") -> "Rotation":
```

</summary>

```py
"""Combine two rotations."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Rotate a vector.</summary>
<details><summary>

```py
    def __matmul__(self, other: Vector) -> Vector:
```

</summary>

```py
"""Rotate a vector."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Rotate a sequence of vectors.</summary>
<details><summary>

```py
    def __matmul__(self, other: Iterable[Vector]) -> Iterable[Vector]:
```

</summary>

```py
"""Rotate a sequence of vectors."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Combine two rotations or rotate a vector or sequence of vectors.</summary>
<details><summary>

```py
    def __matmul__(
        self, other: Union["Rotation", Vector, Iterable[Vector]]
    ) -> Union["Rotation", Vector, Iterable[Vector]]:
```

</summary>

```py
"""Combine two rotations or rotate a vector or sequence of vectors."""
        if isinstance(other, Rotation)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>inverse</code> — Return the inverse rotation such that self @ self.inverse() == Rotation.identity().</summary>
<details><summary>

```py
    def inverse(self) -> "Rotation":
```

</summary>

```py
"""Return the inverse rotation such that self @ self.inverse() == Rotation.identity()."""
        return Rotation.__new__(Rotation, x=-self._x, y=-self._y, z=-self._z, w=self._w)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>angle_to</code> — Calculate the angle to another rotation.</summary>
<details><summary>

```py
    def angle_to(self, other: "Rotation") -> float:
```

</summary>

```py
"""Calculate the angle to another rotation."""
        _, ang = (~self @ other).as_axis_angle()
        return ang

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>axis_angle_to</code> — Calculate the axis and angle to another rotation.</summary>
<details><summary>

```py
    def axis_angle_to(self, other: "Rotation") -> Tuple[Vector, float]:
```

</summary>

```py
"""Calculate the axis and angle to another rotation."""
        ax, ang = (~self @ other).as_axis_angle()
        return self @ ax, ang

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>lerp</code> — Linearly interpolate between two rotations.</summary>
<details><summary>

```py
    def lerp(self, other: "Rotation", f: float) -> "Rotation":
```

</summary>

```py
"""Linearly interpolate between two rotations."""
        ax, ang = (~self @ other).as_axis_angle()
        return self @ Rotation.from_axis_angle(ax, ang * f)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>mean</code> — (no docstring)</summary>
<details><summary>

```py
    def mean(
        rotations: Iterable["Rotation"],
        weights: Optional[Iterable[float]] = None,
        epsilon: float = EPS,
        max_iterations: int = 20,
        return_report: Literal[False] = False,
    ) -> "Rotation": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>mean</code> — (no docstring)</summary>
<details><summary>

```py
    def mean(
        rotations: Iterable["Rotation"],
        weights: Optional[Iterable[float]] = None,
        epsilon: float = EPS,
        max_iterations: int = 20,
        return_report: Literal[True] = True,
    ) -> Tuple["Rotation", RotationMeanReport]: ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>mean</code> — Calculate the mean of a sequence of rotations.
Uses the NASA algorithm (https://ntrs.nasa.gov/citations/20070017872).
Raises when called with an empty sequence or when the sum of weights is zero.
Set return_report to True to get additional information about convergence.</summary>
<details><summary>

```py
    def mean(
        rotations: Iterable["Rotation"],
        weights: Optional[Iterable[float]] = None,
        epsilon: float = EPS,
        max_iterations: int = 20,
        return_report: bool = False,
    ) -> Union["Rotation", Tuple["Rotation", RotationMeanReport]]:
```

</summary>

```py
"""Calculate the mean of a sequence of rotations.
        Uses the NASA algorithm (https://ntrs.nasa.gov/citations/20070017872).
        Raises when called with an empty sequence or when the sum of weights is zero.
        Set return_report to True to get additional information about convergence."""

        weights = weights or repeat(1.0)

        if (
            hasattr(rotations, "__len__")
            and hasattr(weights, "__len__")
            and rotations.__len__() != weights.__len__()
        )
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rotated_towards</code> — Return a rotated version of the rotation such that the local pointer is rotated
towards the global point_along direction. Use interpolate to blend between the two.</summary>
<details><summary>

```py
    def rotated_towards(
        self, pointer: Vector, point_along: Vector, interpolate: float = 1.0
    ) -> "Rotation":
```

</summary>

```py
"""Return a rotated version of the rotation such that the local pointer is rotated
        towards the global point_along direction. Use interpolate to blend between the two.
        """
        current = self @ pointer
        axis = current.cross(point_along)
        if axis.norm() == 0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__eq__</code> — Check if two rotations are equal.</summary>
<details><summary>

```py
    def __eq__(self, other: object) -> bool:
```

</summary>

```py
"""Check if two rotations are equal."""
        if isinstance(other, Rotation)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__invert__</code> — Return the inverse rotation such that r @ ~r == Rotation.identity().</summary>
<details><summary>

```py
    def __invert__(self) -> "Rotation":
```

</summary>

```py
"""Return the inverse rotation such that r @ ~r == Rotation.identity()."""
        return self.inverse()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__str__</code> — Return a string representation of the rotation.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>
<details><summary>

```py
    def __str__(self) -> str:
```

</summary>

```py
"""Return a string representation of the rotation.
        Note that opposite quaternions represent the same rotation and are considered equal
        whereas their string representations are different."""
        return f"±(x={self._x}, y={self._y}, z={self._z}, w={self._w})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__format__</code> — Return a formatted string representation of the rotation;
the format_spec is applied to each element.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>
<details><summary>

```py
    def __format__(self, format_spec: str) -> str:
```

</summary>

```py
"""Return a formatted string representation of the rotation;
        the format_spec is applied to each element.
        Note that opposite quaternions represent the same rotation and are considered equal
        whereas their string representations are different."""

        fx = self._x.__format__(format_spec)
        fy = self._y.__format__(format_spec)
        fz = self._z.__format__(format_spec)
        fw = self._w.__format__(format_spec)
        return f"±(x={fx}, y={fy}, z={fz}, w={fw})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__repr__</code> — Return an eval-able string representation of the rotation.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>
<details><summary>

```py
    def __repr__(self) -> str:
```

</summary>

```py
"""Return an eval-able string representation of the rotation.
        Note that opposite quaternions represent the same rotation and are considered equal
        whereas their string representations are different."""
        return f"Rotation.from_quat(x={self._x}, y={self._y}, z={self._z}, w={self._w})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__hash__</code> — Return a hash of the rotation. Quaternions with opposite signs are equal
and return the same hash.</summary>
<details><summary>

```py
    def __hash__(self) -> int:
```

</summary>

```py
"""Return a hash of the rotation. Quaternions with opposite signs are equal
        and return the same hash."""
        if self._w != 0.0
```

</details>
</details>

</details>

<details><summary><b><code>class Trafo</code></b> — A 3d transformation consisting of a translation and a rotation.</summary>
<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__init__</code> — Create a transformation from a translation and a rotation.</summary>
<details><summary>

```py
    def __init__(self, *, t: Vector = Vector.zero(), r: Rotation = Rotation.identity()):
```

</summary>

```py
"""Create a transformation from a translation and a rotation."""
        # kw_only parameter for @dataclass only supported for Python 3.10+
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "r", r)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>identity</code> — Return the identity transformation.</summary>
<details><summary>

```py
    def identity() -> "Trafo":
```

</summary>

```py
"""Return the identity transformation."""
        return Trafo()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_matrix</code> — Create a transformation from a 3x4 or 4x4 homogeneous transformation matrix.</summary>
<details><summary>

```py
    def from_matrix(
        matrix: Sequence[Sequence[float]],
        row_major: bool = True,
        check_matrix: bool = True,
    ) -> "Trafo":
```

</summary>

```py
"""Create a transformation from a 3x4 or 4x4 homogeneous transformation matrix."""
        if check_matrix
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>as_matrix</code> — Return the transformation as a 3x4 or 4x4 homogeneous transformation matrix.</summary>
<details><summary>

```py
    def as_matrix(
        self, row_major: bool = True, num_rows: Literal[3, 4] = 4
    ) -> List[List[float]]:
```

</summary>

```py
"""Return the transformation as a 3x4 or 4x4 homogeneous transformation matrix."""
        matrix = self.r.as_matrix(row_major=row_major)
        if row_major
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_dh</code> — (no docstring)</summary>
<details><summary>

```py
    def from_dh(
        *, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_dh</code> — (no docstring)</summary>
<details><summary>

```py
    def from_dh(
        *, r: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_dh</code> — (no docstring)</summary>
<details><summary>

```py
    def from_dh(
        *, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, d: float = 0.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_dh</code> — (no docstring)</summary>
<details><summary>

```py
    def from_dh(
        *, r: float = 0.0, alpha: float = 0.0, theta: float = 0.0, d: float = 0.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>from_dh</code> — Create a transformation from Denavit-Hartenberg parameters.
https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
s or d: offset along previous z to the common normal
theta: angle about previous z from old x to new x
r or a: length of the common normal
alpha: angle about common normal, from old z axis to new z axis</summary>
<details><summary>

```py
    def from_dh(
        *,
        a: float = 0.0,  # original letter
        alpha: float = 0.0,
        theta: float = 0.0,
        s: float = 0.0,  # original letter
        r: float = 0.0,  # alternative to a (often used to avoid confusion with alpha)
        d: float = 0.0  # alternative to s (often used since s is used to abbreviate sin)
    ) -> "Trafo":
```

</summary>

```py
"""Create a transformation from Denavit-Hartenberg parameters.
        https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
        s or d: offset along previous z to the common normal
        theta: angle about previous z from old x to new x
        r or a: length of the common normal
        alpha: angle about common normal, from old z axis to new z axis
        """
        if not a == 0.0 and not r == 0.0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>look_at</code> — (no docstring)</summary>
<details><summary>

```py
    def look_at(
        *,
        eye: Vector,
        look_axis: Vector,
        look_at: Vector,
        up_axis: Vector,
        up: Vector,
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>look_at</code> — (no docstring)</summary>
<details><summary>

```py
    def look_at(
        *,
        eye: Vector,
        look_axis: Vector,
        look_along: Vector,
        up_axis: Vector,
        up: Vector,
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>look_at</code> — Create a transformation that looks at a target point.
eye: location
look_axis: local view direction (from eye towards target)
look_at: target if target is a point
look_along: target if target is a direction
up_axis: local up direction
up: global up direction</summary>
<details><summary>

```py
    def look_at(
        *,
        eye: Vector,
        look_axis: Vector,
        look_at: Optional[Vector] = None,
        look_along: Optional[Vector] = None,
        up_axis: Vector,
        up: Vector,
    ) -> "Trafo":
```

</summary>

```py
"""Create a transformation that looks at a target point.
        eye: location
        look_axis: local view direction (from eye towards target)
        look_at: target if target is a point
        look_along: target if target is a direction
        up_axis: local up direction
        up: global up direction
        """

        # rotate the user-defined camera coordinate system (look_axis, up_axis)
        # to align with our convention
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Combine two transformations.</summary>
<details><summary>

```py
    def __matmul__(self, other: "Trafo") -> "Trafo":
```

</summary>

```py
"""Combine two transformations."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Transform a point. (Use myTrafo.r @ myVector to transform a direction).</summary>
<details><summary>

```py
    def __matmul__(self, other: Vector) -> Vector:
```

</summary>

```py
"""Transform a point. (Use myTrafo.r @ myVector to transform a direction)."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Transform a sequence of points.</summary>
<details><summary>

```py
    def __matmul__(self, other: Iterable[Vector]) -> Iterable[Vector]:
```

</summary>

```py
"""Transform a sequence of points."""
        ...

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__matmul__</code> — Combine two transformations or transform a point or sequence of points.
Use myTrafo.r @ myVector to transform a direction.</summary>
<details><summary>

```py
    def __matmul__(
        self, other: Union["Trafo", Vector, Iterable[Vector]]
    ) -> Union["Trafo", Vector, Iterable[Vector]]:
```

</summary>

```py
"""Combine two transformations or transform a point or sequence of points.
        Use myTrafo.r @ myVector to transform a direction."""

        if isinstance(other, Trafo)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>inverse</code> — Return the inverse transformation such that t @ t.inverse() == Trafo.identity().</summary>
<details><summary>

```py
    def inverse(self) -> "Trafo":
```

</summary>

```py
"""Return the inverse transformation such that t @ t.inverse() == Trafo.identity()."""
        inv_r = ~self.r
        return Trafo(t=inv_r @ -self.t, r=inv_r)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>lerp</code> — Linearly interpolate between two transformations.</summary>
<details><summary>

```py
    def lerp(self, other: "Trafo", f: float) -> "Trafo":
```

</summary>

```py
"""Linearly interpolate between two transformations."""
        return Trafo(
            t=self.t.lerp(other.t, f),
            r=self.r.lerp(other.r, f),
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>mean</code> — Calculate the weighted mean of a sequence of transformations.</summary>
<details><summary>

```py
    def mean(
        trafos: Iterable["Trafo"], weights: Optional[Iterable[float]] = None
    ) -> "Trafo":
```

</summary>

```py
"""Calculate the weighted mean of a sequence of transformations."""
        return Trafo(
            t=Vector.mean([t.t for t in trafos], weights),
            r=Rotation.mean([t.r for t in trafos], weights),
        )

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rotated_towards</code> — (no docstring)</summary>
<details><summary>

```py
    def rotated_towards(
        self, pointer: Vector, *, point_at: Vector, interpolate: float = 1.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rotated_towards</code> — (no docstring)</summary>
<details><summary>

```py
    def rotated_towards(
        self, pointer: Vector, *, point_along: Vector, interpolate: float = 1.0
    ) -> "Trafo": ...
:
```

</summary>

```py

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rotated_towards</code> — Return a rotated version of the transformation
such that the local pointer is rotated towards
the global target point (point_at) or direction (point_along).
Use interpolate to blend between current and target.</summary>
<details><summary>

```py
    def rotated_towards(
        self,
        pointer: Vector,
        *,
        point_at: Optional[Vector] = None,
        point_along: Optional[Vector] = None,
        interpolate: float = 1.0
    ) -> "Trafo":
```

</summary>

```py
"""
        Return a rotated version of the transformation
        such that the local pointer is rotated towards
        the global target point (point_at) or direction (point_along).
        Use interpolate to blend between current and target.
        """

        if point_at is not None and point_along is None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__eq__</code> — Check if two transformations are equal.</summary>
<details><summary>

```py
    def __eq__(self, other: object) -> bool:
```

</summary>

```py
"""Check if two transformations are equal."""
        if isinstance(other, Trafo)
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__invert__</code> — Return the inverse transformation such that t @ ~t == Trafo.identity().</summary>
<details><summary>

```py
    def __invert__(self) -> "Trafo":
```

</summary>

```py
"""Return the inverse transformation such that t @ ~t == Trafo.identity()."""
        return self.inverse()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__str__</code> — Return a string representation of the transformation.</summary>
<details><summary>

```py
    def __str__(self) -> str:
```

</summary>

```py
"""Return a string representation of the transformation."""
        return f"(t={self.t}, r={self.r})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__format__</code> — Return a formatted string representation of the transformation.
The format_spec is applied to each element.</summary>
<details><summary>

```py
    def __format__(self, format_spec: str) -> str:
```

</summary>

```py
"""Return a formatted string representation of the transformation.
        The format_spec is applied to each element."""
        ft = self.t.__format__(format_spec)
        fr = self.r.__format__(format_spec)
        return f"(t={ft}, r={fr})"

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__repr__</code> — Return an eval-able string representation of the transformation.</summary>
<details><summary>

```py
    def __repr__(self) -> str:
```

</summary>

```py
"""Return an eval-able string representation of the transformation."""
        return f"Trafo(t={self.t.__repr__()}, r={self.r.__repr__()})"

```

</details>
</details>

</details>

<details><summary><b><code>class Node</code></b> — A node in a tree structure that represents a hierarchy of transformations.</summary>
<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__init__</code> — Create a node with a parent, a transformation in relation to the parent
and a label for debugging and visualizing.</summary>
<details><summary>

```py
    def __init__(self, parent: Union["Node", None], trafo: Trafo, label: str = ""):
```

</summary>

```py
"""Create a node with a parent, a transformation in relation to the parent
        and a label for debugging and visualizing."""
        self._parent = parent
        if parent is not None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>attach_to</code> — Attach the node to a new parent.
If keep_relative_trafo is True, the transformation of the node is updated
to keep the relative transformation to the new parent the same.</summary>
<details><summary>

```py
    def attach_to(
        self, new_parent: Union["Node", None], keep_relative_trafo: bool = False
    ) -> None:
```

</summary>

```py
"""Attach the node to a new parent.
        If keep_relative_trafo is True, the transformation of the node is updated
        to keep the relative transformation to the new parent the same."""

        ancestor = new_parent
        while ancestor is not None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>get_parent</code> — Return the parent node.</summary>
<details><summary>

```py
    def get_parent(self) -> Union["Node", None]:
```

</summary>

```py
"""Return the parent node."""
        return self._parent

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>get_children</code> — Return the child nodes. (Returns a copy of the list that can be modified.)</summary>
<details><summary>

```py
    def get_children(self) -> list["Node"]:
```

</summary>

```py
"""Return the child nodes. (Returns a copy of the list that can be modified.)"""
        return self._children.copy()

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__rshift__</code> — Return the transformation from the Node to another Node in the hierarchy.</summary>
<details><summary>

```py
    def __rshift__(self, other: "Node") -> Trafo:
```

</summary>

```py
"""Return the transformation from the Node to another Node in the hierarchy."""
        if not isinstance(other, Node)
```

</details>
</details>

</details>

<details><summary><b><code>class DebugDrawer</code></b> — Abstract base class for a debug drawer that lets you visualize
vectors, rotations and transformations in relation to each other.
At minimum, the line method must be implemented.</summary>
<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>__init__</code> — Create a debug drawer. The settings are used in the default implementations
of the drawing methods, subclasses are free to ignore them.</summary>
<details><summary>

```py
    def __init__(
        self,
        up: Vector = Vector.ez(),
        arrow_length: float = 1.0,
        font_size: float = 0.1,
        text_direction: Vector = Vector.ex(),
    ):
```

</summary>

```py
"""Create a debug drawer. The settings are used in the default implementations
        of the drawing methods, subclasses are free to ignore them."""
        self.up = up
        self.arrow_length = arrow_length
        self.font_size = font_size
        self.text_direction = text_direction

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>line</code> — Draw a line.</summary>
<details><summary>

```py
    def line(
        self,
        start: Vector,
        end: Vector,
        color: Color,
    ) -> None:
```

</summary>

```py
"""Draw a line."""
        pass

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>arrow</code> — Draw an arrow.</summary>
<details><summary>

```py
    def arrow(self, start: Vector, end: Vector, color: Color) -> None:
```

</summary>

```py
"""Draw an arrow."""
        v = end - start
        if v.norm() > 0
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>point</code> — Draw a point.</summary>
<details><summary>

```py
    def point(self, position: Vector) -> None:
```

</summary>

```py
"""Draw a point."""
        self.line(position - Vector(x=0.01), position + Vector(x=0.01), "default")
        self.line(position - Vector(y=0.01), position + Vector(y=0.01), "default")
        self.line(position - Vector(z=0.01), position + Vector(z=0.01), "default")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>vector</code> — Draw a vector as an arrow from the origin.</summary>
<details><summary>

```py
    def vector(self, vector: Vector) -> None:
```

</summary>

```py
"""Draw a vector as an arrow from the origin."""
        self.arrow(Vector.zero(), vector, "default")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>rotation</code> — Draw a rotation as a rotated coordinate frame.</summary>
<details><summary>

```py
    def rotation(self, rotation: Rotation) -> None:
```

</summary>

```py
"""Draw a rotation as a rotated coordinate frame."""
        o = Vector.zero()
        x = rotation @ Vector.ex()
        y = rotation @ Vector.ey()
        z = rotation @ Vector.ez()
        self.arrow(o, x * self.arrow_length, "x-red")
        self.arrow(o, y * self.arrow_length, "y-green")
        self.arrow(o, z * self.arrow_length, "z-blue")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>trafo</code> — Draw a transformation as a transformed coordinate frame.</summary>
<details><summary>

```py
    def trafo(self, trafo: Trafo) -> None:
```

</summary>

```py
"""Draw a transformation as a transformed coordinate frame."""
        o = trafo.t
        x = trafo.r @ Vector.ex()
        y = trafo.r @ Vector.ey()
        z = trafo.r @ Vector.ez()
        self.arrow(o, o + x * self.arrow_length, "x-red")
        self.arrow(o, o + y * self.arrow_length, "y-green")
        self.arrow(o, o + z * self.arrow_length, "z-blue")

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>text</code> — Draw text at a position.</summary>
<details><summary>

```py
    def text(self, position: Vector, text: str) -> None:
```

</summary>

```py
"""Draw text at a position."""
        # https://en.wikipedia.org/wiki/Fourteen-segment_display

        raster = self.font_size * 0.25  # length of a middle horizontal segment
        # (half width of a character, quarter height of an upper case character)

        # fmt: off
        segment_lines = [
            (0, 4, 2, 4), (2, 4, 2, 2), (2, 2, 2, 0), (2, 0, 0, 0),
            (0, 0, 0, 2), (0, 2, 0, 4), (0, 2, 1, 2), (1, 2, 2, 2),
            (0, 4, 1, 2), (1, 4, 1, 2), (2, 4, 1, 2), (0, 0, 1, 2),
            (1, 0, 1, 2), (2, 0, 1, 2), (1, 0, 1, 0.3),
        ]

        masks = {
            " ": 0x0000, "!": 0x4200, ",": 0x0800, "-": 0x00C0, ".": 0x4000,
            "0": 0x0C3F, "1": 0x0406, "2": 0x00DB, "3": 0x008F, "4": 0x00E6,
            "5": 0x00ED, "6": 0x00FD, "7": 0x1401, "8": 0x00FF, "9": 0x00E7,
            "?": 0x4083, "A": 0x00F7, "B": 0x128F, "C": 0x0039, "D": 0x120F,
            "E": 0x00F9, "F": 0x00F1, "G": 0x00BD, "H": 0x00F6, "I": 0x1209,
            "J": 0x001E, "K": 0x2470, "L": 0x0038, "M": 0x0536, "N": 0x2136,
            "O": 0x003F, "P": 0x00F3, "Q": 0x203F, "R": 0x20F3, "S": 0x018D,
            "T": 0x1201, "U": 0x003E, "V": 0x0C30, "W": 0x2836, "X": 0x2D00,
            "Y": 0x1500, "Z": 0x0C09, "_": 0x0008
        }
        # fmt: on

        x0 = 2.0 * raster
        y0 = -6.0 * raster

        tf = Trafo.look_at(
            eye=position,
            look_axis=Vector.ex(),
            look_along=self.text_direction,
            up_axis=Vector.ey(),
            up=self.up,
        )

        def fsd_char(i: int, mask: int, lower: bool) -> None
```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>node</code> — Draw a node - a Trafo with an arrow from the origin and a label;
origin and Trafo can be shifted by offset.</summary>
<details><summary>

```py
    def node(self, node: Node, offset: Trafo = Trafo()) -> None:
```

</summary>

```py
"""Draw a node - a Trafo with an arrow from the origin and a label;
        origin and Trafo can be shifted by offset."""
        o_parent = offset.t
        o_node = offset @ node.trafo.t
        self.arrow(o_parent, o_node, "default")
        self.trafo(offset @ node.trafo)
        self.text(o_node, node.label)

```

</details>
</details>

<details name='method'><summary>&nbsp;&nbsp;&nbsp;&nbsp;<code>tree</code> — Draw a tree of Trafos starting at the root node, shifted by offset.</summary>
<details><summary>

```py
    def tree(self, root: Node, offset: Trafo = Trafo()) -> None:
```

</summary>

```py
"""Draw a tree of Trafos starting at the root node, shifted by offset."""
        self.node(root, offset)
        for child in root.get_children()
```

</details>
</details>

</details>

<details><summary><b><code>class RotationMeanReport</code></b> — Additional information about convergence of the mean method.</summary>
</details>

