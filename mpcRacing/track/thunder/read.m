clc; clear; close all;
data = load("processed_ThunderHill.mat");

xl = data.fined_left_bound(1,100);
yl = data.fined_left_bound(2,100);

xr = data.fined_right_bound(1,100);
yr = data.fined_right_bound(2,100);



dis = sqrt((xl-xr)^2+(yl-yr)^2);