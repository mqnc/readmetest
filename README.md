$p = \begin{pmatrix}p_x\\p_y\\p_z\end{pmatrix} = (p_x\ p_y\ p_z)^T$

$`p = \left(\begin{matrix}p_x \\\ p_y \\\ p_z\end{matrix}\right) = (p_x\ p_y\ p_z)^T`$

$`p = \begin{pmatrix}p_x\\p_y\\p_z\end{pmatrix} = (p_x\ p_y\ p_z)^T`$

```math
p = \begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix} = (p_x\ p_y\ p_z)^T
```

$$  \begin{bmatrix}
    a \\
    b \\
    c \\
    \end{bmatrix} $$ 

$`\begin{bmatrix}A1 &A2 & A3\\A4 & A5 & A6 \end{bmatrix}`$

```math
\begin{align}
{}^Ap' &= {}^AT_B \cdot {}^Ap \\
\begin{pmatrix}3\\3\end{pmatrix} &= {}^AT_B \cdot \begin{pmatrix}1\\2\end{pmatrix}
\end{align}
```


<details>
<summary><b>Active vs. Passive Transformations</b></summary>

[The distinction between active (or alibi) and passive (or alias) transformations](https://en.wikipedia.org/wiki/Active_and_passive_transformation) can cause lots of bugs and confusion if not clarified and used consistently, [tf2 serves as a negative example here](https://github.com/ros2/geometry2/issues/470).

![active_passive](docs/active_passive.svg)

Active transformation means that transforming a point with a transformation actually moves it in space:

$`  \begin{bmatrix}
    a \\
    b \\
    c \\
    \end{bmatrix} `$

Multiplication with ${}^AT_B$ has actively transformed (moved) the point $p$ from frame $A$ to frame $B$ in the sense that $p'$ has the coordinates in frame $B$ that $p$ had in frame $A$.

Passive transformation on the other hand means that the reference of a point changes from some coordinate system $A$ to some other coordinate system $B$ while the point does not actually move in world coordinates, which is mathematically the opposite operation:

$$
\vec a
$$




Multiplication with ${}^AT_B^{-1}$ has passively transformed (changed the reference of) the point $q$ from $A$ to $B$.

In general, in this project the term "transformation" refers to active transformations if not explicitly stated otherwise.

</details>




![equation](https://github.com/user-attachments/assets/aa59a13a-5cd6-4a98-acf3-07c3005f99ed)




![invert](https://github.com/user-attachments/assets/f32191ca-5fae-4eaf-87f7-a8dc9f586ca6)





