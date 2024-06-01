import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

image_path = 'Image.jpeg'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
yellow_res = cv2.bitwise_and(image, image, mask=yellow_mask)
gray_yellow = cv2.cvtColor(yellow_res, cv2.COLOR_BGR2GRAY)

_, binary_yellow = cv2.threshold(gray_yellow, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

data_points = []
for cnt in contours:
    for point in cnt:
        x, y = point[0]
        data_points.append((x, y))

data_points = sorted(data_points, key=lambda p: p[0])

x_coords = [p[0] for p in data_points]
y_coords = [p[1] for p in data_points]
y_min, y_max = 27, 22

pixel_min_y, pixel_max_y = min(y_coords), max(y_coords)

start_time = datetime.strptime('2024-05-03 09:31', '%Y-%m-%d %H:%M')
end_time = datetime.strptime('2024-05-07 15:48', '%Y-%m-%d %H:%M')

pixel_min_x, pixel_max_x = min(x_coords), max(x_coords)

time_range = end_time - start_time

stock_prices = [y_min + (y_max - y_min) * (y - pixel_min_y) / (pixel_max_y - pixel_min_y) for y in y_coords]
timestamps = [start_time + (time_range * (x - pixel_min_x) / (pixel_max_x - pixel_min_x)) for x in x_coords]

df_with_time = pd.DataFrame({'Timestamp': timestamps, 'Stock_Price': stock_prices})

df_with_time = df_with_time.sample(frac=1)
print(df_with_time)
df_with_time.to_csv('StockPrices.csv', index = False)