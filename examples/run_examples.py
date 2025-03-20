import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

# fig8_ForgedBankNotes.py
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
n_samples = 1000

# figure1_illustration.main()
# figure2_robustness_plots.main()
# figure3_errors_varying_p.main(n_samples, seed)
# figure4_outlier_percentage.main(n_samples, seed)
# figure5_score_comparison.main()
# figure6_performance.main(100, seed)

# figure7_octane.main()
# figure8_ForgedBankNotes.main()
# figure9_topGear.main()
n_samples = 10
table1_and2_error_compaison.main(n_samples, seed)
# table3_batch_size.main(n_samples, seed)
