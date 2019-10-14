try:
    from dlib import rectangle
except ImportError:
    def rectangle(*args, **kwargs):
        raise ImportError


class FaceBox:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.y_min: int = y_min
        self.y_max: int = y_max
        self.x_max: int = x_max
        self.x_min: int = x_min

    def to_dlib_rect(self):
        return rectangle(self.x_min, self.y_min, self.x_max, self.y_max)

    def _to_box(self, target_shape, image_shape):
        image_x_shape, image_y_shape, *_ = image_shape
        x_center, y_center = self.center
        new_x_min, new_x_max = self._calculate_new_expansion(x_center, target_shape, image_x_shape)
        new_y_min, new_y_max = self._calculate_new_expansion(y_center, target_shape, image_y_shape)
        return FaceBox(new_x_min, new_x_max, new_y_min, new_y_max)

    def to_mobile_net_box(self, image_shape):
        mob_net_expansion = 112
        return self._to_box(mob_net_expansion, image_shape)

    def to_quadratic_box(self, image_shape):
        x_diff = self.x_max - self.x_min
        y_diff = self.y_max - self.y_min
        half_expansion = max(x_diff, y_diff) / 2
        return self._to_box(half_expansion, image_shape)

    @property
    def x_size(self):
        return self.x_max - self.x_min

    @property
    def y_size(self):
        return self.y_max - self.y_min


    @staticmethod
    def _calculate_new_expansion(center_value, half_expansion, abs_max):
        # Todo: debug method: incorrect location of BB if a face in the right side of the screen
        new_min = center_value - half_expansion
        new_max = center_value + half_expansion
        if new_min >= 0 and new_max <= abs_max:
            return new_min, new_max
        else:
            if new_min < 0:
                return 0, 2 * half_expansion
            elif new_min >= 0 and new_max > abs_max:
                return abs_max - 2 * half_expansion, abs_max
            else:
                raise ValueError(f"To small room for expansion: {2 * half_expansion} > {abs_max}")

    @property
    def center(self):
        if (self.x_max - self.x_min <= 0) and (self.y_max - self.y_min <= 0):
            # Todo: also log a warning
            return self.x_max, self.y_max
        else:
            return ((self.x_max + self.x_min) // 2,
                    (self.y_max + self.y_min) // 2)

    @property
    def top_left(self):
        return self.x_min, self.y_min

    @property
    def bottom_right(self):
        return self.x_max, self.y_max

    @property
    def x_min(self):
        return self._x_min

    @property
    def x_max(self):
        return self._x_max

    @property
    def y_min(self):
        return self._y_min

    @property
    def y_max(self):
        return self._y_max

    @x_min.setter
    def x_min(self, value):
        self._x_min = int(value)

    @x_max.setter
    def x_max(self, value):
        self._x_max = int(value)

    @y_min.setter
    def y_min(self, value):
        self._y_min = int(value)

    @y_max.setter
    def y_max(self, value):
        self._y_max = int(value)

    @classmethod
    def from_box(cls, box):
        ymin, xmin, ymax, xmax = box
        return cls(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)



