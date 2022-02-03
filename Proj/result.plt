set term png
set output "result.png"
plot 'result_adam.txt' u 1:2 every 4 w l, 'result_rmsprop.txt' u 1:2 every 4 w l,
