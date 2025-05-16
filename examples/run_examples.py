import sys
import os
import time
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

import figure7_octane
import figure8_ForgedBankNotes
import figure9_topGear
import figure1_illustration
import figure2_robustness_plots
import figure3_errors_varying_p 
import figure4_outlier_percentage
import figure5_score_comparison
import figure6_performance
import table1_and2_error_compaison
import table3_batch_size

seed = 0
n_samples = 100

figure1_illustration.main()
figure2_robustness_plots.main() 
# figure3_errors_varying_p.main(n_samples, seed) # takes a long time about 45 min on my machine(10 cores)
# figure4_outlier_percentage.main(n_samples, seed) # takes a long time about 2 min on my machine(10 cores)
figure5_score_comparison.main() 
# figure6_performance.main(100, seed) # takes a long time (many hours)

figure7_octane.main()
figure8_ForgedBankNotes.main()
figure9_topGear.main()
# table1_and2_error_compaison.main(n_samples, seed) # takes a long time (many hours)
# table3_batch_size.main(n_samples, seed) # takes a long time (many hours)
# print("All tables are saved in the tables directory")
print("All figures are saved in the figures directory")

