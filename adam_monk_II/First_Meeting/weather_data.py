# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:16:49 2023

@author: tobia
"""


from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime(2003, 1, 1)
end = datetime(2023, 12, 31)

berlin = Point(52.5200, 13.4050, 70)


data = Daily(berlin, start, end)
data = data.fetch()


