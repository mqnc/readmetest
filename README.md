
<details>
<summary>

__Vector__ — An immutable 3D vector representing a point or a direction.

</summary>


<details>
<summary>
  
__zero__ — Create a zero vector.

</summary>

```py
zero() -> 'Vector'
```

</details>


<details>
<summary>ex — Create a unit vector in x-direction.</summary>

```py
ex() -> 'Vector'
```

</details>


<details>
<summary>ey — Create a unit vector in y-direction.</summary>

```py
ey() -> 'Vector'
```

</details>


<details>
<summary>ez — Create a unit vector in z-direction.</summary>

```py
ez() -> 'Vector'
```

</details>


<details>
<summary>rand_box — Create a random vector with uniform distribution within a box.</summary>

```py
rand_box(min: Sequence[float] = (0.0, 0.0, 0.0), max: Sequence[float] = (1.0, 1.0, 1.0), generator: random.Random = <module 'random' from '/usr/lib/python3.12/random.py'>) -> 'Vector'
```

</details>


<details>
<summary>rand_sphere — Create a random vector with uniform distribution on or in a sphere.</summary>

```py
rand_sphere(radius: float = 1.0, center: Optional[ForwardRef('Vector')] = None, fill: bool = False, generator: random.Random = <module 'random' from '/usr/lib/python3.12/random.py'>) -> 'Vector'
```

</details>


<details>
<summary>__eq__ — Check equality with another vector; true if all elements are equal.</summary>

```py
__eq__(self, other: object) -> bool
```

</details>


<details>
<summary>__ne__ — Check inequality with another vector; true if any element is unequal.</summary>

```py
__ne__(self, other: object) -> bool
```

</details>


<details>
<summary>__add__ — Add another vector element-wise.
Note that this violates LSP for tuples which are expected to concatenate instead.</summary>

```py
__add__(self, other: 'Vector') -> 'Vector'
```

</details>


<details>
<summary>__neg__ — Return the negated vector.</summary>

```py
__neg__(self) -> 'Vector'
```

</details>


<details>
<summary>__sub__ — Subtract another vector.</summary>

```py
__sub__(self, other: 'Vector') -> 'Vector'
```

</details>


<details>
<summary>__mul__ — Multiply by a scalar element-wise.
Note that this violates LSP for tuples which are expected to repeat instead.</summary>

```py
__mul__(self, scalar: float) -> 'Vector'
```

</details>


<details>
<summary>__rmul__ — Multiply by a scalar element-wise.
Note that this violates LSP for tuples which are expected to repeat instead.</summary>

```py
__rmul__(self, scalar: float) -> 'Vector'
```

</details>


<details>
<summary>__truediv__ — Divide by a scalar element-wise.</summary>

```py
__truediv__(self, scalar: float) -> 'Vector'
```

</details>


<details>
<summary>dot — Calculate the dot product with another vector.</summary>

```py
dot(self, other: 'Vector') -> float
```

</details>


<details>
<summary>cross — Calculate the cross product with another vector.</summary>

```py
cross(self, other: 'Vector') -> 'Vector'
```

</details>


<details>
<summary>norm — Calculate the Euclidean norm (length) of the vector.</summary>

```py
norm(self) -> float
```

</details>


<details>
<summary>length — Calculate the length (Euclidean norm) of the vector.</summary>

```py
length(self) -> float
```

</details>


<details>
<summary>normalized — Return a normalized (unit) vector with the same direction; raises when called on a zero vector.</summary>

```py
normalized(self) -> 'Vector'
```

</details>


<details>
<summary>perp — Calculate a vector perpendicular to this vector;
if other is given, the result is perpendicular to both.
Raises when called on a zero vector or when the vectors are parallel.</summary>

```py
perp(self, other: Optional[ForwardRef('Vector')] = None) -> 'Vector'
```

</details>


<details>
<summary>make_basis — Create an orthonormal basis from two vectors;
the direction of the first vector is preserved,
the second is made perpendicular to the first,
the third is perpendicular to the first two.
Raises when called on a zero vector or when the vectors are parallel.</summary>

```py
make_basis(v1: 'Vector', v2: 'Vector') -> Tuple[ForwardRef('Vector'), ForwardRef('Vector'), ForwardRef('Vector')]
```

</details>


<details>
<summary>distance_to — Calculate the Euclidean distance to another vector.</summary>

```py
distance_to(self, other: 'Vector') -> float
```

</details>


<details>
<summary>angle_to — Calculate the angle to another vector.</summary>

```py
angle_to(self, other: 'Vector') -> float
```

</details>


<details>
<summary>lerp — Linearly interpolate between two vectors.</summary>

```py
lerp(self, other: 'Vector', f: float) -> 'Vector'
```

</details>


<details>
<summary>mean — Calculate the weighted mean of a sequence of vectors.
Raises when called with an empty sequence or when the sum of weights is zero.</summary>

```py
mean(vectors: Iterable[ForwardRef('Vector')], weights: Optional[Iterable[float]] = None) -> 'Vector'
```

</details>


<details>
<summary>__str__ — Return a string representation of the vector.</summary>

```py
__str__(self) -> str
```

</details>


<details>
<summary>__format__ — Return a formatted string representation of the vector;
the format_spec is applied to each element.</summary>

```py
__format__(self, format_spec: str) -> str
```

</details>


<details>
<summary>__repr__ — Return an eval-able string representation of the vector.</summary>

```py
__repr__(self) -> str
```

</details>

</details>



<details>
<summary>

### Rotation — An immutable 3D orientation or rotation.

</summary>


<details>
<summary>__setattr__ — (no docstring)</summary>

```py
__setattr__(self, name: str, value: float) -> None
```

</details>


<details>
<summary>__delattr__ — (no docstring)</summary>

```py
__delattr__(self, name: str) -> None
```

</details>


<details>
<summary>__new__ — Construct a rotation from quaternion components without normalization,
intended only for use in classmethods. Use from_quat instead.</summary>

```py
__new__(cls: Type[~_T], *, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> ~_T
```

</details>


<details>
<summary>__init__ — Create the identity rotation.</summary>

```py
__init__(self) -> None
```

</details>


<details>
<summary>identity — Create the identity rotation while being extra explicit about it.</summary>

```py
identity() -> 'Rotation'
```

</details>


<details>
<summary>x — Create a rotation about the x-axis.</summary>

```py
x(angle: float) -> 'Rotation'
```

</details>


<details>
<summary>y — Create a rotation about the y-axis.</summary>

```py
y(angle: float) -> 'Rotation'
```

</details>


<details>
<summary>z — Create a rotation about the z-axis.</summary>

```py
z(angle: float) -> 'Rotation'
```

</details>


<details>
<summary>from_quat — Create a rotation from quaternion components.
Raises an error if the norm deviates from 1 beyond the specified tolerance
or if all components are 0.</summary>

```py
from_quat(*, x: float, y: float, z: float, w: float, tolerance: float = 7.4e-09) -> 'Rotation'
```

</details>


<details>
<summary>as_quat — Return the quaternion components in the specified order.</summary>

```py
as_quat(self, order: Literal['xyzw', 'wxyz']) -> Tuple[float, float, float, float]
```

</details>


<details>
<summary>from_axis_angle — Create a rotation from an axis and an angle.</summary>

```py
from_axis_angle(axis: trafo.trafo.Vector, angle: float) -> 'Rotation'
```

</details>


<details>
<summary>as_axis_angle — Return the axis and angle of the rotation.
The angle is in the range [0, pi).
If the angle is 0, the axis is (1, 0, 0).</summary>

```py
as_axis_angle(self) -> Tuple[trafo.trafo.Vector, float]
```

</details>


<details>
<summary>from_rotvec — Create a rotation from a rotation vector.
(The rotation vector is the axis of rotation scaled by the angle of rotation.)</summary>

```py
from_rotvec(rotvec: trafo.trafo.Vector) -> 'Rotation'
```

</details>


<details>
<summary>as_rotvec — Return the rotation vector of the rotation.
(The rotation vector is the axis of rotation scaled by the angle of rotation.)</summary>

```py
as_rotvec(self) -> trafo.trafo.Vector
```

</details>


<details>
<summary>from_matrix — Create a rotation from a 3x3 rotation matrix.</summary>

```py
from_matrix(matrix: Sequence[Sequence[float]], row_major: bool = True, check_matrix: bool = True) -> 'Rotation'
```

</details>


<details>
<summary>as_matrix — Return the rotation as a 3x3 rotation matrix.</summary>

```py
as_matrix(self, row_major: bool = True) -> List[List[float]]
```

</details>


<details>
<summary>basis — Return the basis vectors of the rotation.</summary>

```py
basis(self) -> Tuple[trafo.trafo.Vector, trafo.trafo.Vector, trafo.trafo.Vector]
```

</details>


<details>
<summary>compose — Compose a rotation from a sequence of rotations about x, y and z.
The sequence is an arbitrarily long string of axis identifiers, e.g. 'XY' or 'zyxZ'.
Use Capital letters for intrinsic rotations (rotate about the new, rotated axes),
use lowercase letters for extrinsic rotations (rotate about the world axes).
Intrinsic and extrinsic rotations can be mixed.</summary>

```py
compose(sequence: str, angles: Sequence[float]) -> 'Rotation'
```

</details>


<details>
<summary>from_euler — Create a rotation from Euler angles. The following orders are allowed:
ZXZ, XYX, YZY, ZYZ, XZX, YXY (proper Euler, intrinsic)
XYZ, YZX, ZXY, XZY, ZYX, YXZ (Tait-Bryan, intrinsic)
zxz, xyx, yzy, zyz, xzx, yxy (proper Euler, extrinsic)
xyz, yzx, zxy, xzy, zyx, yxz (Tait-Bryan, extrinsic)
intrinsic: rotate about the new, rotated axes
extrinsic: rotate about the original axes</summary>

```py
from_euler(order: str, angles: Sequence[float]) -> 'Rotation'
```

</details>


<details>
<summary>as_euler — Return the Euler angles of the rotation. The order is one of:
ZXZ, XYX, YZY, ZYZ, XZX, YXY (proper Euler, intrinsic)
XYZ, YZX, ZXY, XZY, ZYX, YXZ (Tait-Bryan, intrinsic)
zxz, xyx, yzy, zyz, xzx, yxy (proper Euler, extrinsic)
xyz, yzx, zxy, xzy, zyx, yxz (Tait-Bryan, extrinsic)
intrinsic: rotate about the new, rotated axes
extrinsic: rotate about the original axes
In case of a singularity, the first angle is set to 0 for extrinsic rotations,
the third angle is set to 0 for intrinsic rotations.</summary>

```py
as_euler(self, order: str) -> Tuple[float, float, float]
```

</details>


<details>
<summary>from_ypr — Create a rotation from yaw, pitch and roll angles.</summary>

```py
from_ypr(yaw: float, pitch: float, roll: float) -> 'Rotation'
```

</details>


<details>
<summary>as_ypr — Return the yaw, pitch and roll angles of the rotation.
In case of a singularity, roll is set to 0.</summary>

```py
as_ypr(self) -> Tuple[float, float, float]
```

</details>


<details>
<summary>from_rpy — Create a rotation from roll, pitch and yaw angles.</summary>

```py
from_rpy(roll: float, pitch: float, yaw: float) -> 'Rotation'
```

</details>


<details>
<summary>as_rpy — Return the roll, pitch and yaw angles of the rotation.
In case of a singularity, roll is set to 0.</summary>

```py
as_rpy(self) -> Tuple[float, float, float]
```

</details>


<details>
<summary>rand — Create a random rotation with uniform distribution.</summary>

```py
rand(generator: random.Random = <module 'random' from '/usr/lib/python3.12/random.py'>) -> 'Rotation'
```

</details>


<details>
<summary>_rotate_vector — Rotate a vector by the rotation.</summary>

```py
_rotate_vector(self, other: trafo.trafo.Vector) -> trafo.trafo.Vector
```

</details>


<details>
<summary>_rotate_quat — Combine two rotations (quaternion multiplication).</summary>

```py
_rotate_quat(self, other: 'Rotation') -> Tuple[float, float, float, float]
```

</details>


<details>
<summary>__matmul__ — Combine two rotations.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Rotate a vector.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Rotate a sequence of vectors.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Combine two rotations or rotate a vector or sequence of vectors.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Rotation'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>inverse — Return the inverse rotation such that self @ self.inverse() == Rotation.identity().</summary>

```py
inverse(self) -> 'Rotation'
```

</details>


<details>
<summary>angle_to — Calculate the angle to another rotation.</summary>

```py
angle_to(self, other: 'Rotation') -> float
```

</details>


<details>
<summary>axis_angle_to — Calculate the axis and angle to another rotation.</summary>

```py
axis_angle_to(self, other: 'Rotation') -> Tuple[trafo.trafo.Vector, float]
```

</details>


<details>
<summary>lerp — Linearly interpolate between two rotations.</summary>

```py
lerp(self, other: 'Rotation', f: float) -> 'Rotation'
```

</details>


<details>
<summary>mean — (no docstring)</summary>

```py
mean(rotations: Iterable[ForwardRef('Rotation')], weights: Optional[Iterable[float]] = None, epsilon: float = 7.4e-09, max_iterations: int = 20, return_report: bool = False) -> Union[ForwardRef('Rotation'), Tuple[ForwardRef('Rotation'), trafo.trafo.Rotation.RotationMeanReport]]
```

</details>


<details>
<summary>mean — (no docstring)</summary>

```py
mean(rotations: Iterable[ForwardRef('Rotation')], weights: Optional[Iterable[float]] = None, epsilon: float = 7.4e-09, max_iterations: int = 20, return_report: bool = False) -> Union[ForwardRef('Rotation'), Tuple[ForwardRef('Rotation'), trafo.trafo.Rotation.RotationMeanReport]]
```

</details>


<details>
<summary>mean — Calculate the mean of a sequence of rotations.
Uses the NASA algorithm (https://ntrs.nasa.gov/citations/20070017872).
Raises when called with an empty sequence or when the sum of weights is zero.
Set return_report to True to get additional information about convergence.</summary>

```py
mean(rotations: Iterable[ForwardRef('Rotation')], weights: Optional[Iterable[float]] = None, epsilon: float = 7.4e-09, max_iterations: int = 20, return_report: bool = False) -> Union[ForwardRef('Rotation'), Tuple[ForwardRef('Rotation'), trafo.trafo.Rotation.RotationMeanReport]]
```

</details>


<details>
<summary>rotated_towards — Return a rotated version of the rotation such that the local pointer is rotated
towards the global point_along direction. Use interpolate to blend between the two.</summary>

```py
rotated_towards(self, pointer: trafo.trafo.Vector, point_along: trafo.trafo.Vector, interpolate: float = 1.0) -> 'Rotation'
```

</details>


<details>
<summary>__eq__ — Check if two rotations are equal.</summary>

```py
__eq__(self, other: object) -> bool
```

</details>


<details>
<summary>__invert__ — Return the inverse rotation such that r @ ~r == Rotation.identity().</summary>

```py
__invert__(self) -> 'Rotation'
```

</details>


<details>
<summary>__str__ — Return a string representation of the rotation.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>

```py
__str__(self) -> str
```

</details>


<details>
<summary>__format__ — Return a formatted string representation of the rotation;
the format_spec is applied to each element.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>

```py
__format__(self, format_spec: str) -> str
```

</details>


<details>
<summary>__repr__ — Return an eval-able string representation of the rotation.
Note that opposite quaternions represent the same rotation and are considered equal
whereas their string representations are different.</summary>

```py
__repr__(self) -> str
```

</details>


<details>
<summary>__hash__ — Return a hash of the rotation. Quaternions with opposite signs are equal
and return the same hash.</summary>

```py
__hash__(self) -> int
```

</details>

</details>



<details>
<summary>

### Trafo — A 3d transformation consisting of a translation and a rotation.

</summary>


<details>
<summary>__init__ — Create a transformation from a translation and a rotation.</summary>

```py
__init__(self, *, t: trafo.trafo.Vector = Vector(0.0, 0.0, 0.0), r: trafo.trafo.Rotation = Rotation.from_quat(x=0.0, y=0.0, z=0.0, w=1.0))
```

</details>


<details>
<summary>identity — Return the identity transformation.</summary>

```py
identity() -> 'Trafo'
```

</details>


<details>
<summary>from_matrix — Create a transformation from a 3x4 or 4x4 homogeneous transformation matrix.</summary>

```py
from_matrix(matrix: Sequence[Sequence[float]], row_major: bool = True, check_matrix: bool = True) -> 'Trafo'
```

</details>


<details>
<summary>as_matrix — Return the transformation as a 3x4 or 4x4 homogeneous transformation matrix.</summary>

```py
as_matrix(self, row_major: bool = True, num_rows: Literal[3, 4] = 4) -> List[List[float]]
```

</details>


<details>
<summary>from_dh — (no docstring)</summary>

```py
from_dh(*, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0, r: float = 0.0, d: float = 0.0) -> 'Trafo'
```

</details>


<details>
<summary>from_dh — (no docstring)</summary>

```py
from_dh(*, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0, r: float = 0.0, d: float = 0.0) -> 'Trafo'
```

</details>


<details>
<summary>from_dh — (no docstring)</summary>

```py
from_dh(*, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0, r: float = 0.0, d: float = 0.0) -> 'Trafo'
```

</details>


<details>
<summary>from_dh — (no docstring)</summary>

```py
from_dh(*, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0, r: float = 0.0, d: float = 0.0) -> 'Trafo'
```

</details>


<details>
<summary>from_dh — Create a transformation from Denavit-Hartenberg parameters.
https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
s or d: offset along previous z to the common normal
theta: angle about previous z from old x to new x
r or a: length of the common normal
alpha: angle about common normal, from old z axis to new z axis</summary>

```py
from_dh(*, a: float = 0.0, alpha: float = 0.0, theta: float = 0.0, s: float = 0.0, r: float = 0.0, d: float = 0.0) -> 'Trafo'
```

</details>


<details>
<summary>look_at — (no docstring)</summary>

```py
look_at(*, eye: trafo.trafo.Vector, look_axis: trafo.trafo.Vector, look_at: Optional[trafo.trafo.Vector] = None, look_along: Optional[trafo.trafo.Vector] = None, up_axis: trafo.trafo.Vector, up: trafo.trafo.Vector) -> 'Trafo'
```

</details>


<details>
<summary>look_at — (no docstring)</summary>

```py
look_at(*, eye: trafo.trafo.Vector, look_axis: trafo.trafo.Vector, look_at: Optional[trafo.trafo.Vector] = None, look_along: Optional[trafo.trafo.Vector] = None, up_axis: trafo.trafo.Vector, up: trafo.trafo.Vector) -> 'Trafo'
```

</details>


<details>
<summary>look_at — Create a transformation that looks at a target point.
eye: location
look_axis: local view direction (from eye towards target)
look_at: target if target is a point
look_along: target if target is a direction
up_axis: local up direction
up: global up direction</summary>

```py
look_at(*, eye: trafo.trafo.Vector, look_axis: trafo.trafo.Vector, look_at: Optional[trafo.trafo.Vector] = None, look_along: Optional[trafo.trafo.Vector] = None, up_axis: trafo.trafo.Vector, up: trafo.trafo.Vector) -> 'Trafo'
```

</details>


<details>
<summary>__matmul__ — Combine two transformations.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Transform a point. (Use myTrafo.r @ myVector to transform a direction).</summary>

```py
__matmul__(self, other: Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Transform a sequence of points.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>__matmul__ — Combine two transformations or transform a point or sequence of points.
Use myTrafo.r @ myVector to transform a direction.</summary>

```py
__matmul__(self, other: Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]) -> Union[ForwardRef('Trafo'), trafo.trafo.Vector, Iterable[trafo.trafo.Vector]]
```

</details>


<details>
<summary>inverse — Return the inverse transformation such that t @ t.inverse() == Trafo.identity().</summary>

```py
inverse(self) -> 'Trafo'
```

</details>


<details>
<summary>lerp — Linearly interpolate between two transformations.</summary>

```py
lerp(self, other: 'Trafo', f: float) -> 'Trafo'
```

</details>


<details>
<summary>mean — Calculate the weighted mean of a sequence of transformations.</summary>

```py
mean(trafos: Iterable[ForwardRef('Trafo')], weights: Optional[Iterable[float]] = None) -> 'Trafo'
```

</details>


<details>
<summary>rotated_towards — (no docstring)</summary>

```py
rotated_towards(self, pointer: trafo.trafo.Vector, *, point_at: Optional[trafo.trafo.Vector] = None, point_along: Optional[trafo.trafo.Vector] = None, interpolate: float = 1.0) -> 'Trafo'
```

</details>


<details>
<summary>rotated_towards — (no docstring)</summary>

```py
rotated_towards(self, pointer: trafo.trafo.Vector, *, point_at: Optional[trafo.trafo.Vector] = None, point_along: Optional[trafo.trafo.Vector] = None, interpolate: float = 1.0) -> 'Trafo'
```

</details>


<details>
<summary>rotated_towards — Return a rotated version of the transformation
such that the local pointer is rotated towards
the global target point (point_at) or direction (point_along).
Use interpolate to blend between current and target.</summary>

```py
rotated_towards(self, pointer: trafo.trafo.Vector, *, point_at: Optional[trafo.trafo.Vector] = None, point_along: Optional[trafo.trafo.Vector] = None, interpolate: float = 1.0) -> 'Trafo'
```

</details>


<details>
<summary>__eq__ — Check if two transformations are equal.</summary>

```py
__eq__(self, other: object) -> bool
```

</details>


<details>
<summary>__invert__ — Return the inverse transformation such that t @ ~t == Trafo.identity().</summary>

```py
__invert__(self) -> 'Trafo'
```

</details>


<details>
<summary>__str__ — Return a string representation of the transformation.</summary>

```py
__str__(self) -> str
```

</details>


<details>
<summary>__format__ — Return a formatted string representation of the transformation.
The format_spec is applied to each element.</summary>

```py
__format__(self, format_spec: str) -> str
```

</details>


<details>
<summary>__repr__ — Return an eval-able string representation of the transformation.</summary>

```py
__repr__(self) -> str
```

</details>

</details>



<details>
<summary>

### Node — A node in a tree structure that represents a hierarchy of transformations.

</summary>


<details>
<summary>__init__ — Create a node with a parent, a transformation in relation to the parent
and a label for debugging and visualizing.</summary>

```py
__init__(self, parent: Optional[ForwardRef('Node')], trafo: trafo.trafo.Trafo, label: str = '')
```

</details>


<details>
<summary>attach_to — Attach the node to a new parent.
If keep_relative_trafo is True, the transformation of the node is updated
to keep the relative transformation to the new parent the same.</summary>

```py
attach_to(self, new_parent: Optional[ForwardRef('Node')], keep_relative_trafo: bool = False) -> None
```

</details>


<details>
<summary>get_parent — Return the parent node.</summary>

```py
get_parent(self) -> Optional[ForwardRef('Node')]
```

</details>


<details>
<summary>get_children — Return the child nodes. (Returns a copy of the list that can be modified.)</summary>

```py
get_children(self) -> list['Node']
```

</details>


<details>
<summary>__rshift__ — Return the transformation from the Node to another Node in the hierarchy.</summary>

```py
__rshift__(self, other: 'Node') -> trafo.trafo.Trafo
```

</details>

</details>



<details>
<summary>

### DebugDrawer — Abstract base class for a debug drawer that lets you visualize
vectors, rotations and transformations in relation to each other.
At minimum, the line method must be implemented.

</summary>


<details>
<summary>__init__ — Create a debug drawer. The settings are used in the default implementations
of the drawing methods, subclasses are free to ignore them.</summary>

```py
__init__(self, up: trafo.trafo.Vector = Vector(0.0, 0.0, 1.0), arrow_length: float = 1.0, font_size: float = 0.1, text_direction: trafo.trafo.Vector = Vector(1.0, 0.0, 0.0))
```

</details>


<details>
<summary>line — Draw a line.</summary>

```py
line(self, start: trafo.trafo.Vector, end: trafo.trafo.Vector, color: Literal['default', 'x-red', 'y-green', 'z-blue']) -> None
```

</details>


<details>
<summary>arrow — Draw an arrow.</summary>

```py
arrow(self, start: trafo.trafo.Vector, end: trafo.trafo.Vector, color: Literal['default', 'x-red', 'y-green', 'z-blue']) -> None
```

</details>


<details>
<summary>point — Draw a point.</summary>

```py
point(self, position: trafo.trafo.Vector) -> None
```

</details>


<details>
<summary>vector — Draw a vector as an arrow from the origin.</summary>

```py
vector(self, vector: trafo.trafo.Vector) -> None
```

</details>


<details>
<summary>rotation — Draw a rotation as a rotated coordinate frame.</summary>

```py
rotation(self, rotation: trafo.trafo.Rotation) -> None
```

</details>


<details>
<summary>trafo — Draw a transformation as a transformed coordinate frame.</summary>

```py
trafo(self, trafo: trafo.trafo.Trafo) -> None
```

</details>


<details>
<summary>text — Draw text at a position.</summary>

```py
text(self, position: trafo.trafo.Vector, text: str) -> None
```

</details>


<details>
<summary>node — Draw a node - a Trafo with an arrow from the origin and a label;
origin and Trafo can be shifted by offset.</summary>

```py
node(self, node: trafo.trafo.Node, offset: trafo.trafo.Trafo = Trafo(t=Vector(0.0, 0.0, 0.0), r=Rotation.from_quat(x=0.0, y=0.0, z=0.0, w=1.0))) -> None
```

</details>


<details>
<summary>tree — Draw a tree of Trafos starting at the root node, shifted by offset.</summary>

```py
tree(self, root: trafo.trafo.Node, offset: trafo.trafo.Trafo = Trafo(t=Vector(0.0, 0.0, 0.0), r=Rotation.from_quat(x=0.0, y=0.0, z=0.0, w=1.0))) -> None
```

</details>

</details>



<details>
<summary>

### RotationMeanReport — Additional information about convergence of the mean method.

</summary>

</details>

