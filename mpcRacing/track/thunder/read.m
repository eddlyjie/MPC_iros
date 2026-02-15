clc; clear; close all;
data = load("processed_ThunderHill.mat");

x=data.block_info(1,:);
y = data.block_info(2,:);


x0= x(20);
y0 = y(20);
x1= x(21);
y1 = y(21);




dis = sqrt((x0-x1)^2+(y0-y1)^2);