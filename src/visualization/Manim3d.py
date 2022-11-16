from manim import *
import numpy as np
import time


class threeDGraph(ThreeDScene):
    time_step = 0.25
    total_steps = 1200
    time = 0.00
    position_scale = 1
    fade_in = True
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-300,300,20],
            y_range=[-300,300,20],
            z_range=[-300,300,20],
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES, focal_distance=2000)
        self.begin_ambient_camera_rotation()
        text3d = Text("150 Body Simulation")
        self.add_fixed_in_frame_mobjects(text3d)
        text3d.to_corner(UL)
        positions = self.load_positions_from_csv(self.time)
        dots = []
        body_count = positions.shape[0]
        print(body_count)
        print('Creating Dots...')
        for i in range(body_count):
            mass = positions[i, 3]
            point = axes.coords_to_point(positions[i, 0], positions[i, 1], positions[i, 2])
            dot = Dot3D(point=point, radius=mass/420, color=GREEN).move_to(point)
            dots.append(dot)
        print('Done creating dots...')

        if self.fade_in:
            animations=[FadeIn(axes), FadeIn(VGroup(*dots))]
            self.play(AnimationGroup(*animations, lag_ratio=0.75))
        else:
            self.add(axes, VGroup(*dots))
        self.time += self.time_step

        while self.time<self.total_steps*self.time_step:
            start = time.time()
            positions = self.load_positions_from_csv(self.time)

            animations = []
            for i in range(body_count):
                new_point = axes.coords_to_point(positions[i,0], positions[i,1], positions[i,2])
                animations.append(dots[i].animate(run_time=0.1).move_to(new_point))

            print('Playing animations...')
            self.play(AnimationGroup(*animations))
            end = time.time()
            print('One step took', end-start, 'seconds to render.')
            self.time+=self.time_step

    def load_positions_from_csv(self, timestamp : float):
        s = '%.2f' % timestamp
        filename = "exampleResults/"+str(s)+".csv"
        print('Loading File', filename, '...')
        return np.loadtxt(filename, delimiter=',', dtype=np.longdouble)/self.position_scale