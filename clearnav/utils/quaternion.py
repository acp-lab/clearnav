import torch as th


class Quaternion:
    """
    Quaternion class implemented in pytorch
    follows a similar interface to scipy.spatial.transform.Rotation
    """

    def __init__(self, w=None, x=None, y=None, z=None, num=1, device=th.device("cpu")):
        assert (
            type(w) == type(x) == type(y) == type(z)
        )  # "w, x, y, z should have the same type"
        if w is None:
            self.w = th.ones(num, device=device)
            self.x = th.zeros(num, device=device)
            self.y = th.zeros(num, device=device)
            self.z = th.zeros(num, device=device)
        elif isinstance(w, (int, float)):
            self.w = th.ones(num, device=device) * w
            self.x = th.ones(num, device=device) * x
            self.y = th.ones(num, device=device) * y
            self.z = th.ones(num, device=device) * z
        elif isinstance(w, th.Tensor):
            assert w.ndimension() == 1
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        else:
            raise ValueError("unsupported type")

    @staticmethod
    def from_euler(roll, pitch, yaw, order="zyx"):
        """
        roll, pitch, yaw: th.Tensor (N,)
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Tait%E2%80%93Bryan_angles
        if order == "zyx":
            # abbreviations
            cy = th.cos(yaw * 0.5)
            sy = th.sin(yaw * 0.5)
            cp = th.cos(pitch * 0.5)
            sp = th.sin(pitch * 0.5)
            cr = th.cos(roll * 0.5)
            sr = th.sin(roll * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return Quaternion(w, x, y, z)
        else:
            raise NotImplementedError

    @staticmethod
    def from_rotvec(rotvec):
        """
        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation
        https://www.mathworks.com/help/fusion/ref/quaternion.rotvec.html

        rotvec (N, 3)
        """
        assert rotvec.ndimension() == 2 and rotvec.shape[0] == 3

        angle = th.norm(rotvec, dim=0)

        # avoid division by zero (when rotation vector has zero magnitude)
        unit_vec = (rotvec / angle).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        w = th.cos(angle * 0.5)
        x = unit_vec[0, :] * th.sin(angle * 0.5)
        y = unit_vec[1, :] * th.sin(angle * 0.5)
        z = unit_vec[2, :] * th.sin(angle * 0.5)
        return Quaternion(w, x, y, z)

    def to(self, device):
        self.w = self.w.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.z = self.z.to(device)
        return self

    def to_tensor(self, scalar_first=True) -> th.Tensor:
        """
        if scalar_first: (w, x, y, z); else: (x, y, z, w)
        """
        if scalar_first:
            return th.stack([self.w, self.x, self.y, self.z])
        else:
            return th.stack([self.x, self.y, self.z, self.w])

    @property
    def shape(self):
        return 4, len(self)

    @property
    def x_axis(self):
        # first column of R (3,N)
        w, x, y, z = self.w, self.x, self.y, self.z
        return th.stack(
            [1 - 2 * (y**2 + z**2), 2 * (x * y + z * w), 2 * (x * z - y * w)]
        )

    @property
    def y_axis(self):
        # second column of R (3,N)
        w, x, y, z = self.w, self.x, self.y, self.z
        return th.stack(
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)]
        )

    @property
    def z_axis(self):
        # third column of R (3,N)
        w, x, y, z = self.w, self.x, self.y, self.z
        return th.stack(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
        )

    @property
    def real(self):
        return self.w

    @property
    def imag(self):
        return th.stack([self.x, self.y, self.z])

    def as_matrix(self):
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
        w, x, y, z = self.w, self.x, self.y, self.z
        return th.stack(
            [
                th.stack(
                    [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)]
                ),
                th.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)]
                ),
                th.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
                ),
            ]
        )

    def as_euler(self, order="zyx"):
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
        if order == "zyx":
            w, x, y, z = self.w, self.x, self.y, self.z
            roll = th.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
            pitch = th.asin(2 * (w * y - x * z))
            yaw = th.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
            return th.stack([roll, pitch, yaw])
        else:
            raise NotImplementedError

    def clone(self):
        return Quaternion(
            self.w.clone(), self.x.clone(), self.y.clone(), self.z.clone()
        )

    def detach(self):
        return Quaternion(
            self.w.detach(), self.x.detach(), self.y.detach(), self.z.detach()
        )

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        # The Hamilton product defines the multiplication of two quaternions
        # https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
        if isinstance(other, Quaternion):
            w = (
                self.w * other.w
                - self.x * other.x
                - self.y * other.y
                - self.z * other.z
            )
            x = (
                self.w * other.x
                + self.x * other.w
                + self.y * other.z
                - self.z * other.y
            )
            y = (
                self.w * other.y
                - self.x * other.z
                + self.y * other.w
                + self.z * other.x
            )
            z = (
                self.w * other.z
                + self.x * other.y
                - self.y * other.x
                + self.z * other.w
            )
            return Quaternion(w, x, y, z)
        else:
            raise ValueError("unsupported type")

    def __len__(self):
        try:
            return len(self.w)
        except TypeError:
            return 1

    def __repr__(self):
        return f"({self.w}, {self.x}i, {self.y}j, {self.z}k)"


def unit_test():
    """
    Tests functionality of Quaternion class
    1. Constructing quaternions
    2. Applying sequential rotations
    3. Converting quaternion <-> euler angles
    4. Converting quaternion <-> axis angles
    """
    from scipy.spatial.transform import Rotation as R

    q = Quaternion()
    assert q.shape == (4, 1)
    assert th.allclose(q.x_axis[0], th.tensor([1.0, 0.0, 0.0]))
    assert th.allclose(q.y_axis[0], th.tensor([0.0, 1.0, 0.0]))
    assert th.allclose(q.z_axis[0], th.tensor([0.0, 0.0, 1.0]))

    q = Quaternion(num=8)
    assert q.shape == (4, 8)
    assert q.to_tensor().shape == (4, 8)

    # use Rotation to generate and apply 100 random rotations sequentially
    rots = R.random(num=100)
    q_final = Quaternion()
    ref_soln = R.identity()
    for rot in rots:
        x, y, z, w = rot.as_quat()  # scalar-last
        q_final = q_final * Quaternion(w, x, y, z)
        ref_soln = ref_soln * rot

    # check final quaternion matches reference solution
    q_final = q_final.to_tensor(scalar_first=False)
    ref_soln = th.from_numpy(ref_soln.as_quat()).to(q_final.dtype).reshape(4, 1)
    assert th.allclose(q_final, ref_soln, atol=1e-6), f"\n{ref_soln}\n{q_final}"

    # generate euler angles from (-pi/2, pi/2)
    input_euler = 0.49 * th.pi * (2 * th.rand(3, 10000) - 1)
    q = Quaternion.from_euler(*input_euler)
    output_euler = q.as_euler(order="zyx")
    assert th.allclose(input_euler, output_euler, atol=1e-4), abs(
        input_euler - output_euler
    )

    # check that 0,0,0 rotvec is the unit quaternion
    q = Quaternion.from_rotvec(th.tensor([[0.0, 0.0, 0.0]]).T)
    assert th.allclose(q.to_tensor(), Quaternion().to_tensor())

    # generate random axis angles
    rotvec = th.rand(3, 100)
    q = Quaternion.from_rotvec(rotvec).to_tensor(scalar_first=False)
    ref_soln = th.from_numpy(R.from_rotvec(rotvec.T).as_quat()).to(q.dtype).T
    assert th.allclose(q, ref_soln)


if __name__ == "__main__":
    unit_test()
