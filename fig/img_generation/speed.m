parsing = 2.9e7
times = [1.7e6 14540 280670 2.65e6]
pie(times)
title("Comparison of execution time", "fontsize", 13)
legend({"Line Segmentation" "Pile detector" "RANSAC detector" "Pattern fitting"}, "fontsize", 13)