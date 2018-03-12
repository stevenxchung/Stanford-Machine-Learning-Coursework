% This is where I test all the ML stuff

clc
clear all

%% Initial Conditions

A = magic(10);
x = transpose(1:10);

%% Loops

v = zeros(10, 1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) + A(i, j) * x(j);
  end
end

%% Vectorize

v1 = A * x;
%v2 = Ax;
v3 = A.*x;
v4 = sum(A*x);

%% Testing Disp

disp(v3)
disp(v3(:, 1))
