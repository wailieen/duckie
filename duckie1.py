from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2 as cv

class DontCrushDuckieTaskSolution(TaskSolution):
	def __init__(self, generated_task):
		super().__init__(generated_task)

	def solve(self):
		env = self.generated_task['env']
		img, _, _, _ = env.step([0,0])
		
		condition = True
		
		while condition:
			img, reward, done, info = env.step([1, 0])
			
			img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
			lower_range = np.array([0, 190, 250])
			upper_range = np.array([1, 220, 255])

			mask = cv.inRange(img_rgb, lower_range, upper_range)
			contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

			x, y, w, h = cv.boundingRect(contours[0])
			if (h > 100):
				condition = False
				img, reward, done, info = env.step([0, 0])
			else:
				condition = True
			env.render()