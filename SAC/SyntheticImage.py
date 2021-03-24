import numpy as np
from scipy import interpolate

class SyntheticImage:
    "This class is responsible for generating a synthtic image used for deep reinforcement learning applications."
    "The objective of the class is to return a synthetic image based upon only GPS track vertices sampled at distinct frequency."

    def __init__(self, env = None, car_shape = (4,8), image_shape = (96,96), render_distance = 30, road_width = 40/6, fill = True, interpolate = True, obstacles = False):
        "The environment and environmental parameters are based upon the OPENAI GYM : "
        "https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py"

        self.env = env
        self.fill = fill
        self.car_shape = car_shape
        self.image_shape = image_shape
        self.render_distance = render_distance
        self.road_width = road_width
        self.interpolate = interpolate
        self.image = np.zeros((image_shape[0], image_shape[1]))
        self.obstacles = obstacles

    def generate_image(self):
        "This function collaborates with all other functions and is responsible for generating all components of the image."

        self.image = np.zeros((self.image_shape[0], self.image_shape[1]))   # reset the image
        
        road_success = self.synthesize_road()
        if not road_success:
            print('There was a problem generating the road')

        if self.obstacles:
            obstacle_success = self.synthesize_obstacles()
            if not obstacle_success:
                print('There was a problem generating the obstacles')

        car_success = self.display_car()
        if not car_success:
            print('There was a problem displaying the car')

        return self.image

    def fill_image(self, inner, outer, color = 1):
        " The fill_image function is responsible for filling the image whilst being given an inside and outer layer of the parameters"
        " This can be used for either the road, obstacles, or alternative vehicles."
        " The color is the color to which the object must be colored since we are only doing a 2D monotonic image. "

        inner_discrete = np.array(np.round(inner), dtype = int) + np.array([int(self.image_shape[0]/2), int(self.image_shape[1]/2)]) #This ensures coordinates are integer based due to pixel's discrete nature
        outer_discrete = np.array(np.round(outer), dtype = int) + np.array([int(self.image_shape[0]/2), int(self.image_shape[1]/2)]) #This ensures coordinates are integer based due to pixel's discrete nature

        ix_outside_inner = np.where(inner_discrete >= self.image_shape[0])[0]                                    # So this is within the image
        ix_outside_outer = np.where(outer_discrete >= self.image_shape[1])[0]                                    # So this is within the image
        ix_inner = np.where(inner_discrete < 0)[0]                                                               # this zero is for axis = 0
        ix_outer = np.where(outer_discrete < 0)[0]                                                               # this zero is for axis = 0
        ix = np.unique(np.hstack((ix_inner,ix_outer,ix_outside_inner,ix_outside_outer)))                         # We combine all the indexes which need to be deleted
        inner_filtered = np.delete(inner_discrete, ix, axis = 0)                                                 # Here we delete for both the inner and the outer
        outer_filtered = np.delete(outer_discrete, ix, axis = 0)                                                 # Here we delete for both the inner and the outer

        self.image[inner_filtered[:,0], inner_filtered[:,1]] = color                                             # coloring in the actual points we have, before the full fill.
        self.image[outer_filtered[:,0], outer_filtered[:,1]] = color

        xrange = (outer_filtered[:,0] - inner_filtered[:,0])                                                    # Here we define the xrange from which the outer (bigger) to inner (smaller)
        yrange = (outer_filtered[:,1] - inner_filtered[:,1])
        delta_x = np.round(xrange)
        delta_y = np.round(yrange)

        if self.fill:
            for i in range(inner_filtered.shape[0]):  
                im_x_range = np.sign(delta_x[i])*np.arange(abs(delta_x[i])) + inner_filtered[i,0]
                im_y_range = np.sign(delta_y[i])*np.arange(abs(delta_y[i])) + inner_filtered[i,1]
                self.image[im_x_range, outer_filtered[i,1]] = color
                self.image[inner_filtered[i,0], im_y_range] = color 
        
        return True

    def display_car(self):
        "This function simply displays the car on the image"

        car = np.array([int(self.image_shape[0]/2), int(self.image_shape[1]/2)]) - self.car_shape + np.array([3,3])
        self.image[car[0]:(car[0] + self.car_shape[0]), car[1]:(car[1] + self.car_shape[1])] = 0.5
        return True

    def synthesize_road(self):
        " The synthesize_road function essentially plots the road on the image - "
        " It is vital that this is executed first, because obstacles and other vehicles must go on top of this."

        beta = np.array(self.env.track)[:,1]                                                                  # This is the angle of the road at a given vertex
        center_road = np.array([np.array(self.env.track)[:,2] , np.array(self.env.track)[:,3]]).T             # These are the coordinates from the environment of the 
        outer_road = center_road + np.array([self.road_width*np.cos(beta), self.road_width*np.sin(beta)]).T   # The outer road is classified as the right hand side of the road once initially rendered        
        inner_road = center_road - np.array([self.road_width*np.cos(beta), self.road_width*np.sin(beta)]).T   # The inner road is classifed as the left hand side of the road once initially rendered
        car_pos = np.array([self.env.car.hull.position[0], self.env.car.hull.position[1]])                    # This is the position of the vehicle at any given step
                                                                
        # centering shifting axis system so car is at (0,0)
        inner_road_c = inner_road - car_pos
        outer_road_c = outer_road - car_pos

        # applying the rotation matrix to convert from global to local coordinate system
        t_angle = -self.env.car.hull.angle
        R = lambda x,y: np.array([x*np.cos(t_angle) - y*np.sin(t_angle), y*np.cos(t_angle) + x*np.sin(t_angle)]).T
        inner_transformed = R(inner_road_c[:,0], inner_road_c[:,1])
        outer_transformed = R(outer_road_c[:,0], outer_road_c[:,1])

        # finding the coordinates of the n_nearest coordinates
        near_road_pos = (center_road[:,0] - car_pos[0])**2 + (center_road[:,1] - car_pos[1])**2    # calculates the distance of the road coordinates to the car position
        ix_near_road_pos = np.argsort(near_road_pos)                                               # sort the distances so the smallest is in front e.g. [0.3, 2.4, 12.9 ...]
        image_coordinates = ix_near_road_pos[0:self.render_distance]                               # we obtain the self.render_distance nearest coordinates e.g. if 20 = 10 behind, 10 ahead
        image_coordinates = np.sort(image_coordinates)[::-1]                                       # now we sort the coordinates in the descending order [88,87,86,10,9,3] so we have continuity
        difference_check = np.diff(image_coordinates)                                              # we use the difference check for the beginning stage because the usual diff = 1 ; e.g. diff > 10 we know.

        if np.max(abs(difference_check)) > self.render_distance:
            ix_change = np.argmax(abs(difference_check))                                           # We find the location where the change has occurred [87,86,10,0] = 1
            image_coordinates[:ix_change + 1] = np.sort(image_coordinates[:ix_change + 1])         # Here we are organizing such that the order is such that the flow of the coordinates is smooth
            image_coordinates[ix_change + 1:] = np.sort(image_coordinates[ix_change + 1:])         # Here we are organizing such that the order is such that the flow of the coordinates is smooth
            y_inner_sorted = inner_transformed[image_coordinates,1]
            x_inner_sorted = inner_transformed[image_coordinates,0]
            y_outer_sorted = outer_transformed[image_coordinates,1]
            x_outer_sorted = outer_transformed[image_coordinates,0]
        else:
            y_inner_sorted = inner_transformed[image_coordinates,1][::-1]
            x_inner_sorted = inner_transformed[image_coordinates,0][::-1]
            y_outer_sorted = outer_transformed[image_coordinates,1][::-1]
            x_outer_sorted = outer_transformed[image_coordinates,0][::-1]

        if self.interpolate:
            x_int_inner = self.hexoplate_interpolate(x_inner_sorted)
            y_int_inner = self.hexoplate_interpolate(y_inner_sorted)
            x_int_outer = self.hexoplate_interpolate(x_outer_sorted)
            y_int_outer = self.hexoplate_interpolate(y_outer_sorted)
        else:
            self.indicator_outer = x_inner_sorted
            self.indicator_inner = y_inner_sorted
            x_int_inner = x_inner_sorted
            y_int_inner = y_inner_sorted
            x_int_outer = x_outer_sorted
            y_int_outer = y_outer_sorted

        inner_road = np.array([x_int_inner,y_int_inner]).T
        outer_road = np.array([x_int_outer,y_int_outer]).T

        success = self.fill_image(inner_road, outer_road, color = 1)
        return success

    def synthesize_obstacles(self):
        "This function generates and plots the obstacles found in the environment"

        # Centering the obstacles coordinates relative to the car's position.
        car_pos = np.array([self.env.car.hull.position[0], self.env.car.hull.position[1]])
        obstacles_positions_c = np.zeros((self.env.obstacles_pos.shape)) 
        obstacles_positions_c[:,:,0] = self.env.obstacles_pos[:,:,0] - car_pos[0]
        obstacles_positions_c[:,:,1] = self.env.obstacles_pos[:,:,1] - car_pos[1]

        # rotating the obstacles coordinates relative to the vehicle's
        t_angle = -self.env.car.hull.angle
        obstacles_transformed = np.zeros((self.env.obstacles_pos.shape)) 
        obstacles_transformed[:,:,0] = obstacles_positions_c[:,:,0]*np.cos(t_angle) - obstacles_positions_c[:,:,1]*np.sin(t_angle)
        obstacles_transformed[:,:,1] = obstacles_positions_c[:,:,1]*np.cos(t_angle) + obstacles_positions_c[:,:,0]*np.sin(t_angle)

        # choosing which obstacle to render
        nearest_obstacle_distance = np.sum((self.env.obstacles_pos[:,:,0] - car_pos[0])**2, axis = 1) + np.sum((self.env.obstacles_pos[:,:,1] - car_pos[1])**2, axis = 1)
        chosen_obstacle = np.argmin(nearest_obstacle_distance)
        obstacle_x = obstacles_transformed[chosen_obstacle,:,0]
        obstacle_y = obstacles_transformed[chosen_obstacle,:,1]

        # Just remember that the order is [1, 0]//[2, 3]
        x_obstacle_inner = self.octoplate_interpolate(np.array([obstacle_x[3], obstacle_x[0]])) 
        y_obstacle_inner = self.octoplate_interpolate(np.array([obstacle_y[3], obstacle_y[0]])) 
        x_obstacle_outer = self.octoplate_interpolate(np.array([obstacle_x[2], obstacle_x[1]])) 
        y_obstacle_outer = self.octoplate_interpolate(np.array([obstacle_y[2], obstacle_y[1]])) 

        inner_obstacle = np.array([x_obstacle_inner, y_obstacle_inner]).T
        outer_obstacle = np.array([x_obstacle_outer, y_obstacle_outer]).T
        
        self.fill_image(inner_obstacle, outer_obstacle, color = 0.1)
        return True

    def hexoplate_interpolate(self, A):
        "This is my own interpolation function which aims to reduce the distance between the points of the road."
        "This function interpolaates between only two points in a 1D approach"
        "This is exceptionally usefull since it is independent of which side is in front first"

        A_new = np.zeros((A.shape[0] - 1, 6))
        for i in range(len(A) - 1):
            diff = (A[i] - A[i+1])/6
            A_new[i] = np.array([A[i], A[i] - diff, A[i] - diff*2, A[i] - diff*3, A[i] - diff*4, A[i] - diff*5])
        return np.ndarray.flatten(A_new)

    def octoplate_interpolate(self,A):
        "This is my own interpolation function which aims to reduce the distance between the points of the road."
        "This function interpolaates between only two points in a 1D approach"
        "This is exceptionally usefull since it is independent of which side is in front first"

        A_new = np.zeros((A.shape[0] - 1, 9))
        for i in range(len(A) - 1):
            diff = (A[i] - A[i+1])/8
            A_new[i] = np.array([A[i], A[i] - diff, A[i] - diff*2, A[i] - diff*3, A[i] - diff*4, A[i] - diff*5, A[i] - diff*6, A[i] - diff*7, A[i+1]])
        return np.ndarray.flatten(A_new)